# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List
from transformers import AutoConfig
import torch
import time
import psutil

from parrot.utils import RecyclePool, get_logger, time_counter_in_nanoseconds
from parrot.sampling_config import SamplingConfig

from .model_instantiation import instantiate_model
from .mem import init_model_cache_storage
from ..context.block_context import BlockContext
from .iter_state import IterationState
from ..context.context_manager import EngineContextManager
from ..primitive_job import PrimitiveJob, Fill, Generate
from ..config import BuiltinConfig


logger = get_logger("BuiltinRunner")


def _format_mib(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024:.2f} MiB"


def get_model_memory(model) -> float:
    model_mem = 0
    for param in model.parameters():
        model_mem += param.nelement() * param.element_size()
    for buffer in model.buffers():
        model_mem += buffer.nelement() * buffer.element_size()
    return model_mem / 1024 / 1024


def get_kv_cache_block_size_bytes(hf_config, builtin_config: BuiltinConfig) -> int:
    num_layers = hf_config.num_hidden_layers
    block_size = builtin_config.block_size
    num_heads = hf_config.num_attention_heads
    head_size = hf_config.hidden_size // num_heads
    dtype_size = torch.empty((), dtype=builtin_config.dtype).element_size()
    return num_layers * block_size * num_heads * head_size * dtype_size * 2


class BuiltinRunner:
    """Minimal Builtin LLM Runner with adaption to Parrot."""

    def __init__(self, model_name: str, config: BuiltinConfig):
        self.builtin_config = config
        self.context_manager = EngineContextManager()

        # Init CUDA env
        if self.builtin_config.device.type == "cuda":
            self.local_rank = (
                self.builtin_config.device.index
                if self.builtin_config.device.index is not None
                else 0
            )
            torch.cuda.set_device(self.local_rank)
        else:
            self.local_rank = 0

        # Load Model
        self.hf_model_config = AutoConfig.from_pretrained(model_name)

        # Override max seq len
        if self.builtin_config.max_seq_len is not None:
            self.hf_model_config.max_position_embeddings = (
                self.builtin_config.max_seq_len
            )

        self.model = instantiate_model(
            model_name, self.hf_model_config, self.builtin_config
        )
        self.model_mem = get_model_memory(self.model)
        logger.info(f"Model memory usage: {self.model_mem:.2f} MiB.")

        self._maybe_configure_kv_cache_blocks()
        self.kv_cache_manager = RecyclePool(
            "KVCache pool", pool_size=self.builtin_config.num_kv_cache_blocks
        )

        # Init model cache storage
        init_model_cache_storage(self.hf_model_config, self.builtin_config)

    def _maybe_configure_kv_cache_blocks(self) -> None:
        if self.builtin_config.gpu_memory_utilization is None:
            logger.info(
                f"Using configured KV cache blocks: "
                f"{self.builtin_config.num_kv_cache_blocks}."
            )
            return

        if self.builtin_config.device.type != "cuda":
            raise ValueError(
                "gpu_memory_utilization is only supported for CUDA devices."
            )

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_memory, total_memory = torch.cuda.mem_get_info(self.local_rank)
        device_name = torch.cuda.get_device_name(self.local_rank)
        used_memory = total_memory - free_memory
        target_memory = int(total_memory * self.builtin_config.gpu_memory_utilization)
        available_for_kv = target_memory - used_memory
        block_size_bytes = get_kv_cache_block_size_bytes(
            self.hf_model_config, self.builtin_config
        )

        if available_for_kv <= 0:
            raise RuntimeError(
                "No GPU memory is available for KV cache under "
                f"gpu_memory_utilization={self.builtin_config.gpu_memory_utilization}. "
                f"total={_format_mib(total_memory)}, used={_format_mib(used_memory)}, "
                f"target={_format_mib(target_memory)}."
            )

        num_blocks = int(available_for_kv // block_size_bytes)
        if num_blocks <= 0:
            raise RuntimeError(
                "GPU memory available for KV cache is smaller than one block. "
                f"available={_format_mib(available_for_kv)}, "
                f"block_size={_format_mib(block_size_bytes)}."
            )

        previous_blocks = self.builtin_config.num_kv_cache_blocks
        self.builtin_config.num_kv_cache_blocks = num_blocks
        logger.info(
            "Auto-configured KV cache blocks from gpu_memory_utilization: "
            f"device={device_name}, "
            f"utilization={self.builtin_config.gpu_memory_utilization}, "
            f"total={_format_mib(total_memory)}, free={_format_mib(free_memory)}, "
            f"used={_format_mib(used_memory)}, target={_format_mib(target_memory)}, "
            f"available_for_kv={_format_mib(available_for_kv)}, "
            f"block_size={_format_mib(block_size_bytes)}, "
            f"num_kv_cache_blocks={num_blocks} "
            f"(configured={previous_blocks})."
        )

    @torch.inference_mode()
    def run_iter(self, jobs: List[PrimitiveJob]) -> (int, int):
        logger.debug(f"Running {len(jobs)} jobs. ")

        # torch.cuda.synchronize()
        st = time_counter_in_nanoseconds()

        # We should sort jobs such that Fill jobs are before Generation jobs.
        jobs.sort(key=lambda job: isinstance(job, Generate))

        # Some generation jobs should do "first sampling"
        first_sampling_states: List[torch.Tensor] = []
        first_sampling_config: List[SamplingConfig] = []
        first_sampling_jobs: List[Generate] = []

        # Allocate new context blocks
        for job in jobs:
            # NOTE(chaofan): if we use engine, this is not necessary.
            if job.context is None:
                self.context_manager.bind_job_context(
                    job,
                    BlockContext,
                    block_size=self.builtin_config.block_size,
                    kv_cache_manager=self.kv_cache_manager,
                )

            # Allocate blocks
            allocated_blocks_id: List[int] = []

            if isinstance(job, Fill):
                job.context.token_ids.extend(job.token_ids)
                job.context.allocate(len(job.token_ids))
            elif isinstance(job, Generate):
                job.context.allocate(1)
                last_hidden_state = job.context.get_last_hidden_state()
                if last_hidden_state is not None:
                    first_sampling_states.append(last_hidden_state)
                    first_sampling_config.append(job.sampling_config)
                    first_sampling_jobs.append(job)
                    job.context.last_hidden_state = None

            job.context.token_kv_block_ids.extend(allocated_blocks_id)

        # First sampling
        if len(first_sampling_states) > 0:
            logger.debug(
                f"Running first sampling for {len(first_sampling_states)} jobs."
            )
            first_sampling_states = torch.stack(first_sampling_states)
            first_sampling_tokens = (
                self.model.sampler(first_sampling_states, first_sampling_config)
                .cpu()
                .tolist()
            )
            for i, job in enumerate(first_sampling_jobs):
                job.put_token(first_sampling_tokens[i])

        # Prepare iteration state
        iteration_state = IterationState(
            jobs,
            self.hf_model_config,
            self.builtin_config,
        )

        # Convert inputs
        input_ids = []
        input_positions = []

        for job in jobs:
            context_len = job.context.get_context_len()
            if isinstance(job, Fill):
                input_ids.extend(job.token_ids)
                input_positions.extend(
                    range(context_len - len(job.token_ids), context_len)
                )
            elif isinstance(job, Generate):
                input_ids.append(job.context.get_last_token_id())
                input_positions.append(context_len - 1)

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.int64,
            device=self.builtin_config.device,
        )
        input_positions = torch.tensor(
            input_positions,
            dtype=torch.int64,
            device=self.builtin_config.device,
        )

        torch.cuda.synchronize()
        st_model = time_counter_in_nanoseconds()

        # Execute model
        fill_hidden_states, next_tokens = self.model(
            input_ids, input_positions, iteration_state
        )

        next_tokens = next_tokens.cpu().tolist()

        torch.cuda.synchronize()
        ed_model = time_counter_in_nanoseconds()

        torch.cuda.empty_cache()  # Release unactivated GPU memory

        assert fill_hidden_states.shape[0] + len(next_tokens) == len(jobs)

        model_time = ed_model - st_model

        # Update context
        for i, job in enumerate(jobs):
            assert job.context is not None, "Context should be assigned."
            if isinstance(job, Fill):
                job.context.last_hidden_state = fill_hidden_states[i]
                job.finish_event.set()
            elif isinstance(job, Generate):
                token_id = next_tokens[i - iteration_state.num_fill_jobs]
                job.put_token(token_id)
                if job.check_stop():
                    job.finish_event.set()

        ed = time_counter_in_nanoseconds()

        e2e_time = ed - st
        logger.debug(
            f"Finished running {len(jobs)} jobs. "
            f"({iteration_state.num_fill_jobs} Fills, {iteration_state.num_generation_jobs} Generations). "
            f"Total Time used: {e2e_time / 1e6} (ms); "
            f"Model Time used: {model_time / 1e6} (ms)."
        )

        return e2e_time, model_time
