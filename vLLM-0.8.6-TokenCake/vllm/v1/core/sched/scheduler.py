# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from typing import Optional, Union

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.logger import init_logger
from vllm.mcp.agent_info import AgentInfoManager
from vllm.mcp.agent_offload_coordinator import (AgentOffloadCoordinator,
                                                PressureSnapshot)
from vllm.mcp.agent_scheduler import AgentScheduler
from vllm.mcp.mcp_manager import MCPFunctionManager
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.utils import cdiv
from vllm.v1.core.cpu_offloading_kv_cache_manager import (
    CpuOffloadingKVCacheManager)
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.request_queue import (RequestQueue, SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%r, fallback to %s", name, value,
                       default)
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r, fallback to %s", name,
                       value, default)
        return default


MCP_KV_OFFLOAD_MIN_GPU_USAGE = _env_float(
    "VLLM_MCP_MIN_GPU_USAGE_FOR_OFFLOAD",
    _env_float("VLLM_AGENT_GPU_USAGE_HIGH_WATERMARK", 0.6),
)
MCP_KV_OFFLOAD_HIGH_PRESSURE_GPU_USAGE = _env_float(
    "VLLM_MCP_HIGH_PRESSURE_GPU_USAGE",
    max(MCP_KV_OFFLOAD_MIN_GPU_USAGE + 0.05, 0.85),
)
MCP_KV_OFFLOAD_SCORE_THRESHOLD = _env_float("VLLM_MCP_OFFLOAD_SCORE_THRESHOLD",
                                            1.0)
MCP_KV_OFFLOAD_RELIEF_GUARD_BLOCKS = max(
    0, _env_int("VLLM_MCP_OFFLOAD_RELIEF_GUARD_BLOCKS", 1))
MCP_KV_OFFLOAD_BACKOFF_STEPS = max(
    1, _env_int("VLLM_MCP_OFFLOAD_BACKOFF_STEPS", 8))
MCP_KV_OFFLOAD_EVICTION_WINDOW_BLOCKS = max(
    1, _env_int("VLLM_MCP_OFFLOAD_EVICTION_WINDOW_BLOCKS", 128))
MCP_KV_OFFLOAD_MAX_RELIEF_BLOCKS = max(
    1, _env_int("VLLM_MCP_OFFLOAD_MAX_RELIEF_BLOCKS", 32))
AGENT_PREEMPTION_SCORE_MARGIN = 3000.0
AGENT_PRESSURE_PREFILL_TOKEN_CAP = 256
AGENT_PRESSURE_GPU_USAGE = 0.80
TEMPORAL_SELECTION_POLICY = os.getenv("VLLM_TEMPORAL_SELECTION_POLICY",
                                      "first_fit").lower()


class Scheduler(SchedulerInterface):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager

        self.mcp_manager = MCPFunctionManager()
        logical_block_bytes = sum(
            len(group.layer_names) * group.kv_cache_spec.page_size_bytes
            for group in kv_cache_config.kv_cache_groups)
        self.mcp_manager.configure_transfer_layout(logical_block_bytes)
        self.agent_info_manager = AgentInfoManager()

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.include_finished_set = include_finished_set

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

        self.num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert self.num_gpu_blocks is not None and self.num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size
        self.max_token_num = self.num_gpu_blocks * self.block_size

        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            self.connector = KVConnectorFactory.create_connector_v1(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        # req_id -> Request
        self.requests: dict[str, Request] = {}

        self.policy = SchedulingPolicy.get_policy(self.scheduler_config.policy)
        logger.info(f"[scheduler] scheduling policy: {self.policy}")

        # Priority queues for requests.
        self.waiting: RequestQueue = create_request_queue(
            self.policy, self.agent_info_manager)
        self.running: list[Request] = []
        self.scheduled_req_ids: set[str] = set()
        self.preempted_stop_requests: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        # Request id -> CachedRequestData
        self._cached_reqs_data: dict[str, CachedRequestData] = {}

        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=self.scheduler_config,
            mm_registry=mm_registry,
        )
        self.max_num_encoder_input_tokens = encoder_compute_budget
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        self.num_lookahead_tokens = 0
        speculative_config = vllm_config.speculative_config
        if speculative_config and speculative_config.method == "eagle":
            self.num_lookahead_tokens = (
                speculative_config.num_speculative_tokens)

        self.enable_kvcache_cpu_offloading = self.cache_config.enable_suncake
        logger.info(
            f"[scheduler] enable_kvcache_cpu_offloading: {self.enable_kvcache_cpu_offloading}"
        )
        if self.enable_kvcache_cpu_offloading:
            self.kv_cache_manager = CpuOffloadingKVCacheManager(
                kv_cache_config=kv_cache_config,
                max_model_len=self.max_model_len,
                enable_caching=self.cache_config.enable_prefix_caching,
                caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
                log_stats=self.log_stats,
            )
        else:
            self.kv_cache_manager = KVCacheManager(
                kv_cache_config=kv_cache_config,
                max_model_len=self.max_model_len,
                enable_caching=self.cache_config.enable_prefix_caching,
                caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
                log_stats=self.log_stats,
            )

        self.enable_agent_scheduling = self.cache_config.enable_suncake
        self.agent_scheduler = AgentScheduler(self.enable_agent_scheduling,
                                              self.agent_info_manager,
                                              self.num_gpu_blocks,
                                              self.block_size,
                                              self.kv_cache_manager)
        self.agent_offload_coordinator = AgentOffloadCoordinator(
            min_gpu_usage_for_offload=MCP_KV_OFFLOAD_MIN_GPU_USAGE,
            high_pressure_gpu_usage=MCP_KV_OFFLOAD_HIGH_PRESSURE_GPU_USAGE,
            offload_score_threshold=MCP_KV_OFFLOAD_SCORE_THRESHOLD,
        )
        logger.info(
            "[scheduler] offload coordinator thresholds: min_gpu_usage=%.3f high_pressure=%.3f score_threshold=%.3f",
            MCP_KV_OFFLOAD_MIN_GPU_USAGE,
            MCP_KV_OFFLOAD_HIGH_PRESSURE_GPU_USAGE,
            MCP_KV_OFFLOAD_SCORE_THRESHOLD,
        )
        self._current_pressure_snapshot: Optional[PressureSnapshot] = None
        self._offload_rejection_backoff: dict[str, tuple[str, tuple, int]] = {}
        self._waiting_h2d_prefetches: dict[str, dict[str, set[int]]] = {}

    def has_requests(self) -> bool:
        return self.has_unfinished_requests() or self.has_finished_requests()

    def _select_preemption_victim(self,
                                  request: Optional[Request] = None
                                  ) -> Request:
        if self.policy == SchedulingPolicy.AGENT:
            get_agent_priority = getattr(getattr(self, "waiting", None),
                                         "get_agent_priority", None)
            if get_agent_priority is not None:
                victim = max(
                    self.running,
                    key=lambda r:
                    (-float(get_agent_priority(r)), r.arrival_time),
                )
                if request is not None and victim is not request:
                    request_score = float(get_agent_priority(request))
                    victim_score = float(get_agent_priority(victim))
                    if request_score < victim_score + AGENT_PREEMPTION_SCORE_MARGIN:
                        return request
                return victim
            return max(
                self.running,
                key=lambda r: (-r.priority, r.arrival_time),
            )
        if self.policy == SchedulingPolicy.PRIORITY:
            return max(
                self.running,
                key=lambda r: (r.priority, r.arrival_time),
            )
        return self.running[-1]

    def _remove_waiting_request(self, request: Request) -> None:
        self.waiting.remove_request(request)

    def _cap_agent_prefill_tokens(self, request: Request,
                                  num_new_tokens: int) -> int:
        if (getattr(self, "policy", None) != SchedulingPolicy.AGENT
                or not getattr(self, "enable_agent_scheduling", False)
                or num_new_tokens <= AGENT_PRESSURE_PREFILL_TOKEN_CAP):
            return num_new_tokens

        has_backlog = bool(self.waiting) or len(self.running) >= max(
            2, self.max_num_running_reqs // 2)
        if not has_backlog and self.kv_cache_manager.usage < AGENT_PRESSURE_GPU_USAGE:
            return num_new_tokens
        return AGENT_PRESSURE_PREFILL_TOKEN_CAP

    def _ensure_agent_offload_coordinator(self) -> AgentOffloadCoordinator:
        if not hasattr(self, "agent_offload_coordinator"):
            self.agent_offload_coordinator = AgentOffloadCoordinator(
                min_gpu_usage_for_offload=MCP_KV_OFFLOAD_MIN_GPU_USAGE,
                high_pressure_gpu_usage=MCP_KV_OFFLOAD_HIGH_PRESSURE_GPU_USAGE,
                offload_score_threshold=MCP_KV_OFFLOAD_SCORE_THRESHOLD,
            )
        if not hasattr(self, "_current_pressure_snapshot"):
            self._current_pressure_snapshot = None
        if not hasattr(self, "_offload_rejection_backoff"):
            self._offload_rejection_backoff = {}
        return self.agent_offload_coordinator

    def _get_agent_priority_for_request(self, request: Request) -> float:
        if request is None:
            return 0.0
        get_agent_priority = getattr(getattr(self, "waiting", None),
                                     "get_agent_priority", None)
        if get_agent_priority is None:
            return float(getattr(request, "priority", 0) or 0)
        try:
            return float(get_agent_priority(request))
        except Exception:
            return float(getattr(request, "priority", 0) or 0)

    def _get_request_for_coordination(self,
                                      request_id: str) -> Optional[Request]:
        request = getattr(self, "requests", {}).get(request_id)
        if request is not None:
            return request
        for candidate in getattr(self, "running", []):
            if candidate.request_id == request_id:
                return candidate
        for candidate in getattr(self, "waiting", []):
            if candidate.request_id == request_id:
                return candidate
        return None

    def _get_coordination_metadata(self, request_id: str) -> dict:
        request = self._get_request_for_coordination(request_id)
        if request is not None:
            return dict(request.agent_info or {})
        return dict(self.mcp_manager.request_metadata.get(request_id, {}))

    def _refresh_pressure_snapshot(self) -> PressureSnapshot:
        coordinator = self._ensure_agent_offload_coordinator()
        block_pool = getattr(self.kv_cache_manager, "block_pool", None)
        gpu_usage = float(getattr(self.kv_cache_manager, "usage", 0.0) or 0.0)
        num_gpu_blocks = int(getattr(self, "num_gpu_blocks", 0) or 0)
        if num_gpu_blocks <= 0:
            num_gpu_blocks = max(
                1,
                int(getattr(self, "max_token_num", 0) or 0) //
                max(1, self.block_size))
        gpu_free_blocks = (block_pool.get_num_free_blocks()
                           if block_pool is not None
                           and hasattr(block_pool, "get_num_free_blocks") else
                           max(0, int(num_gpu_blocks * (1.0 - gpu_usage))))
        gpu_clean_free_blocks = gpu_free_blocks
        cpu_free_blocks = (block_pool.get_num_free_cpu_blocks() if (
            getattr(self, "enable_kvcache_cpu_offloading", False)
            and block_pool is not None
            and hasattr(block_pool, "get_num_free_cpu_blocks")) else 0)
        snapshot = coordinator.build_pressure_snapshot(
            gpu_total_blocks=num_gpu_blocks,
            gpu_free_blocks=gpu_free_blocks,
            gpu_usage=gpu_usage,
            gpu_clean_free_blocks=gpu_clean_free_blocks,
            waiting_requests=getattr(self, "waiting", []),
            block_size=self.block_size,
            cpu_free_blocks=cpu_free_blocks,
            agent_scheduler=self.agent_scheduler if getattr(
                self, "enable_agent_scheduling", False) else None,
            mcp_manager=self.mcp_manager,
        )
        self._current_pressure_snapshot = snapshot
        self.mcp_manager.record_coordination_event(
            "pressure_snapshot",
            step_id=snapshot.step_id,
            gpu_usage=snapshot.gpu_usage,
            gpu_free_blocks=snapshot.gpu_free_blocks,
            gpu_clean_free_blocks=snapshot.gpu_clean_free_blocks,
            waiting_demand_blocks=snapshot.waiting_demand_blocks,
            critical_waiting_demand_blocks=snapshot.
            critical_waiting_demand_blocks,
            upload_debt_blocks=snapshot.upload_debt_blocks,
            cpu_free_blocks=snapshot.cpu_free_blocks,
            recent_swap_events=snapshot.recent_swap_events,
        )
        return snapshot

    def _get_pressure_snapshot(self) -> PressureSnapshot:
        if getattr(self, "_current_pressure_snapshot", None) is None:
            return self._refresh_pressure_snapshot()
        return self._current_pressure_snapshot

    def _manage_mcp_kv_cache(self, *, process_offload: bool = True):
        self.kv_cache_manager.block_pool.clear_step_d2h_swap_map()
        self.kv_cache_manager.block_pool.clear_step_h2d_swap_map()

        if (not self.mcp_manager.has_requests()
                and not self.mcp_manager.offloaded_request_data):
            self._current_pressure_snapshot = None
            return

        if (self.mcp_manager.requests_to_offload
                and not self.mcp_manager.offloaded_request_data
                and not self.mcp_manager.requests_to_upload
                and not self.waiting):
            self._current_pressure_snapshot = None
            self.mcp_manager.increment_coordination_counter(
                "offload_fast_gate_skip", reason="no-waiting-request")
            return

        coordinator = self._ensure_agent_offload_coordinator()
        coordinator.begin_step()
        self._current_pressure_snapshot = None

        self._refresh_pressure_snapshot()

        self._manage_mcp_kv_cache_upload()
        if process_offload:
            self._manage_mcp_kv_cache_offload()

        if (process_offload
                and self.kv_cache_manager.block_pool.step_d2h_swap_map):
            logger.info(
                "[scheduler] offloaded %s KV blocks to CPU",
                len(self.kv_cache_manager.block_pool.step_d2h_swap_map),
            )
        if self.kv_cache_manager.block_pool.step_h2d_swap_map:
            logger.info(
                "[scheduler] uploaded %s KV blocks to GPU",
                len(self.kv_cache_manager.block_pool.step_h2d_swap_map),
            )

    def _manage_mcp_kv_cache_upload(self):
        if not self.enable_kvcache_cpu_offloading:
            return

        requests_to_upload = self.mcp_manager.get_requests_to_upload()
        if not requests_to_upload:
            return

        for req_id in requests_to_upload:
            if self.mcp_manager.request_transfer_states.get(
                    req_id) == "D2H_PENDING":
                self.mcp_manager.increment_coordination_counter(
                    "upload_deferred", reason="d2h-pending")
                continue
            offloaded_cpu_blocks = self.mcp_manager.get_offloaded_request_data(
                req_id)
            if not offloaded_cpu_blocks:
                self.mcp_manager.mark_request_uploaded(req_id)
                continue

            released_cpu_blocks: list[KVCacheBlock] = []
            seen_cpu_block_ids: set[int] = set()
            for cpu_block in offloaded_cpu_blocks:
                if cpu_block.block_id in seen_cpu_block_ids:
                    continue
                seen_cpu_block_ids.add(cpu_block.block_id)
                released_cpu_blocks.append(cpu_block)

            self.kv_cache_manager.block_pool.free_cpu_blocks(
                released_cpu_blocks)
            self.mcp_manager.record_coordination_event(
                "offload_released_to_cpu_cache",
                req_id,
                reason="demand-driven-restore",
                released_blocks=len(released_cpu_blocks),
            )
            self.mcp_manager.cancel_request_transfers(req_id)
            self.mcp_manager.mark_request_uploaded(req_id)
            pressure = self._get_pressure_snapshot()
            pressure.cpu_free_blocks = min(
                self.kv_cache_manager.block_pool.num_cpu_blocks,
                pressure.cpu_free_blocks + len(released_cpu_blocks),
            )

    def _waiting_request_admission(
        self,
        request: Request,
        *,
        max_relief_blocks: int,
        pressure: PressureSnapshot,
    ) -> tuple[bool, int, int, str]:
        if request.status not in (RequestStatus.WAITING,
                                  RequestStatus.PREEMPTED):
            return False, 0, 0, "dependency-not-ready"
        if len(getattr(self, "running",
                       ())) >= getattr(self, "max_num_running_reqs", 1 << 30):
            return False, 0, 0, "sequence-capacity-full"

        computed_blocks = []
        computed_cpu_blocks = []
        num_computed_tokens = request.num_computed_tokens
        peek_computed_blocks = getattr(self.kv_cache_manager,
                                       "peek_computed_blocks", None)
        if peek_computed_blocks is not None:
            (computed_blocks, computed_cpu_blocks,
             num_computed_tokens) = peek_computed_blocks(request)

        num_new_tokens = max(0, request.num_tokens - num_computed_tokens)
        token_budget = getattr(self, "max_num_scheduled_tokens",
                               num_new_tokens)
        if num_new_tokens <= 0 or token_budget <= 0:
            return False, 0, 0, "token-budget-empty"
        num_new_tokens = min(num_new_tokens, token_budget)
        num_new_tokens = self._cap_agent_prefill_tokens(
            request, num_new_tokens)

        if self.enable_agent_scheduling:
            demand = self.agent_scheduler.estimate_incremental_gpu_blocks(
                request,
                num_new_tokens,
                new_computed_blocks=computed_blocks,
                new_computed_cpu_blocks=computed_cpu_blocks,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )
            if not self.agent_scheduler.can_schedule_req(
                    request,
                    num_new_tokens,
                    computed_blocks,
                    computed_cpu_blocks,
                    num_lookahead_tokens=self.num_lookahead_tokens):
                return False, demand, 0, "agent-reservation-blocked"
        else:
            req_blocks = self.kv_cache_manager.req_to_blocks[
                request.request_id]
            unique_gpu_hits = [
                block for block in computed_blocks if block not in req_blocks
            ]
            total_tokens = (request.num_computed_tokens +
                            (len(unique_gpu_hits) + len(computed_cpu_blocks)) *
                            self.block_size + num_new_tokens +
                            self.num_lookahead_tokens)
            demand = max(
                0,
                cdiv(total_tokens, self.block_size) - len(req_blocks) -
                len(unique_gpu_hits),
            )

        required_relief = max(0, demand - pressure.gpu_free_blocks)
        if required_relief <= 0:
            return False, demand, 0, "already-admissible-without-offload"
        if required_relief > max_relief_blocks:
            return False, demand, required_relief, "insufficient-relief"
        return True, demand, required_relief, "admissible-after-relief"

    def _find_best_fit_waiting_request(
        self,
        eviction_window_blocks: int,
        pressure: Optional[PressureSnapshot] = None,
    ) -> tuple[Optional[Request], int, int]:
        if eviction_window_blocks <= 0:
            return None, 0, 0
        pressure = pressure or self._get_pressure_snapshot()

        candidates: list[tuple[Request, int]] = []
        for request in getattr(self, "waiting", ()):
            _, demand, _, reason = self._waiting_request_admission(
                request,
                max_relief_blocks=0,
                pressure=pressure,
            )
            if reason == "already-admissible-without-offload" and demand > 0:
                candidates.append((request, demand))

        if not candidates:
            return None, 0, 0

        def rank(item: tuple[Request, int]) -> tuple[float, int, float]:
            request, demand = item
            priority = (self._get_agent_priority_for_request(request)
                        if self.enable_agent_scheduling else float(
                            request.priority))
            if TEMPORAL_SELECTION_POLICY == "priority_first":
                return priority, demand, -request.arrival_time
            if TEMPORAL_SELECTION_POLICY == "best_fit":
                return (-abs(eviction_window_blocks - demand), priority,
                        -request.arrival_time)
            return demand, priority, -request.arrival_time

        if TEMPORAL_SELECTION_POLICY != "first_fit":
            candidates.sort(key=rank, reverse=True)

        sequence_slots = max(
            0,
            getattr(self, "max_num_running_reqs", len(candidates)) -
            len(getattr(self, "running", ())),
        )
        token_block_budget = cdiv(
            max(0, getattr(self, "max_num_scheduled_tokens", 0)),
            self.block_size,
        )
        remaining_blocks = min(
            eviction_window_blocks,
            pressure.gpu_free_blocks,
            token_block_budget
            if token_block_budget > 0 else eviction_window_blocks,
        )
        selected_request: Optional[Request] = None
        aggregate_demand = 0
        selected_count = 0
        for request, demand in candidates:
            if selected_count >= sequence_slots or remaining_blocks <= 0:
                break
            if demand > remaining_blocks:
                continue
            if selected_request is None:
                selected_request = request
            aggregate_demand += demand
            remaining_blocks -= demand
            selected_count += 1

        if selected_request is None:
            return None, 0, 0
        return selected_request, aggregate_demand, aggregate_demand

    def _offload_pressure_signature(self, pressure: PressureSnapshot) -> tuple:
        return (
            pressure.gpu_free_blocks,
            pressure.gpu_clean_free_blocks,
            pressure.cpu_free_blocks,
            pressure.waiting_request_count,
            pressure.waiting_demand_blocks,
            pressure.upload_debt_blocks,
        )

    def _select_reusable_eviction_frontier(
        self,
        imminent_eviction_ids: set[int],
        excluded_gpu_block_ids: set[int],
        max_blocks: int,
    ) -> list[KVCacheBlock]:
        block_pool = self.kv_cache_manager.block_pool
        ranked_candidates: list[tuple[int, int, KVCacheBlock]] = []
        seen_hashes: set[BlockHashType] = set()
        block = block_pool.free_block_queue.free_list_head
        while block is not None:
            block_hash = block.block_hash
            if (block.block_id in imminent_eviction_ids
                    and block.block_id not in excluded_gpu_block_ids
                    and block_hash is not None
                    and block_hash not in seen_hashes):
                seen_hashes.add(block_hash)
                reuse_score = block_pool.get_prefix_reuse_score(block_hash)
                if (reuse_score >= 2
                        and block_pool.get_ineffective_preservation_count(
                            block_hash) == 0):
                    ranked_candidates.append(
                        (reuse_score, block_pool.get_prefix_depth(block_hash),
                         block))
            block = block.next_free_block

        ranked_candidates.sort(key=lambda item: (-item[0], item[1]))
        provisional = ranked_candidates[:max(0, int(max_blocks))]
        selected_hashes = {
            block.block_hash
            for _, _, block in provisional if block.block_hash is not None
        }

        def ancestors_remain_reachable(block_hash: BlockHashType) -> bool:
            known, parent = block_pool.get_prefix_parent(block_hash)
            if not known:
                return False
            while parent is not None:
                if parent in selected_hashes:
                    known, parent = block_pool.get_prefix_parent(parent)
                    if not known:
                        return False
                    continue
                if block_pool.peek_cpu_shadow_block(parent) is not None:
                    known, parent = block_pool.get_prefix_parent(parent)
                    if not known:
                        return False
                    continue
                gpu_copies = tuple(
                    block_pool.cached_block_hash_to_block.get(parent,
                                                              {}).values())
                if not gpu_copies or all(copy.block_id in imminent_eviction_ids
                                         for copy in gpu_copies):
                    return False
                known, parent = block_pool.get_prefix_parent(parent)
                if not known:
                    return False
            return True

        selected = [
            block for _, _, block in provisional
            if block.block_hash is not None
            and ancestors_remain_reachable(block.block_hash)
        ]
        if not selected and ranked_candidates:
            self.mcp_manager.increment_coordination_counter(
                "offload_fast_gate_skip", reason="unreachable-prefix-frontier")
        return selected

    def _plan_offload_request(
        self,
        req_id: str,
        req_blocks_to_offload: list[KVCacheBlock],
        eviction_plan: Optional[tuple[Optional[Request], int, int]] = None,
    ) -> tuple[bool, int, Optional[str]]:
        pressure = self._get_pressure_snapshot()
        signature = self._offload_pressure_signature(pressure)
        coordinator = self._ensure_agent_offload_coordinator()
        backoff = self._offload_rejection_backoff.get(req_id)
        step_id = coordinator.step_id
        if (backoff is not None and backoff[1] == signature
                and step_id < backoff[2]):
            self.mcp_manager.increment_coordination_counter(
                "offload_backoff_skip", reason=backoff[0])
            return False, 0, None

        block_pool = getattr(self.kv_cache_manager, "block_pool", None)
        restore_feedback = (
            block_pool.get_restore_feedback() if block_pool is not None
            and hasattr(block_pool, "get_restore_feedback") else {})

        if eviction_plan is None:
            eviction_plan = self._find_best_fit_waiting_request(
                MCP_KV_OFFLOAD_EVICTION_WINDOW_BLOCKS, pressure)
        fit_request, fit_demand, expected_evictions = eviction_plan
        requested_preservation = min(
            len(req_blocks_to_offload),
            expected_evictions,
            MCP_KV_OFFLOAD_MAX_RELIEF_BLOCKS,
        ) if fit_request is not None else 0
        evaluation_blocks = (requested_preservation if requested_preservation
                             > 0 else len(req_blocks_to_offload))
        predicted_duration = self.mcp_manager.get_remaining_predicted_duration(
            req_id)
        d2h_transfer_time = self.mcp_manager.estimate_transfer_time(
            requested_preservation, direction="d2h")
        h2d_transfer_time = self.mcp_manager.estimate_transfer_time(
            requested_preservation, direction="h2d")
        transfer_time = d2h_transfer_time + h2d_transfer_time
        cpu_free_blocks = (block_pool.get_num_free_cpu_blocks()
                           if block_pool is not None
                           and hasattr(block_pool, "get_num_free_cpu_blocks")
                           else requested_preservation)
        use_agent_importance = getattr(self, "enable_agent_scheduling", False)
        request = self._get_request_for_coordination(req_id)
        metadata = self._get_coordination_metadata(req_id)
        decision = self._ensure_agent_offload_coordinator().evaluate_offload(
            request_id=req_id,
            freed_blocks=evaluation_blocks,
            cpu_free_blocks=cpu_free_blocks,
            predicted_duration=predicted_duration,
            transfer_time=transfer_time,
            queue_fit_request=fit_request,
            queue_fit_demand_blocks=fit_demand,
            pressure=pressure,
            request=request,
            metadata=metadata,
            agent_scheduler=self.agent_scheduler
            if use_agent_importance else None,
            priority_getter=self._get_agent_priority_for_request
            if use_agent_importance else None,
        )

        self.mcp_manager.record_offload_decision(
            req_id,
            allowed=decision.allowed,
            reason=decision.reason,
            predicted_duration=predicted_duration,
            transfer_time=transfer_time,
            queue_fit_request_id=decision.queue_fit_request_id,
            freed_blocks=requested_preservation,
            eligible_suffix_blocks=len(req_blocks_to_offload),
            gpu_usage=pressure.gpu_usage,
            intended_admission_request_id=decision.queue_fit_request_id,
            intended_admission_demand_blocks=fit_demand,
            expected_eviction_blocks=expected_evictions,
            preserved_eviction_blocks=requested_preservation,
            required_relief_blocks=0,
            safety_guard_blocks=0,
            final_score=decision.final_score,
            stall_margin=decision.stall_margin,
            pressure_score=decision.pressure_score,
            fit_score=decision.fit_score,
            importance_penalty=decision.importance_penalty,
            completion_penalty=decision.completion_penalty,
            upload_safety_score=decision.upload_safety_score,
            cpu_capacity_score=decision.cpu_capacity_score,
            churn_penalty=decision.churn_penalty,
            queue_fit_demand_blocks=decision.queue_fit_demand_blocks,
            cpu_free_blocks=decision.cpu_free_blocks,
            request_importance=decision.request_importance.__dict__,
            pressure_step_id=pressure.step_id,
            gpu_free_blocks=pressure.gpu_free_blocks,
            gpu_clean_free_blocks=pressure.gpu_clean_free_blocks,
            waiting_demand_blocks=pressure.waiting_demand_blocks,
            critical_waiting_demand_blocks=pressure.
            critical_waiting_demand_blocks,
            upload_debt_blocks=pressure.upload_debt_blocks,
            recent_swap_events=pressure.recent_swap_events,
            restore_feedback=restore_feedback,
            observed_reuse_blocks=sum(
                int(block_pool.get_prefix_reuse_score(block.block_hash))
                for block in req_blocks_to_offload
                if block.block_hash is not None),
            hot_candidate_blocks=sum(
                int(block_pool.get_prefix_reuse_score(block.block_hash) > 0)
                for block in req_blocks_to_offload
                if block.block_hash is not None),
        )
        if decision.allowed:
            self._offload_rejection_backoff.pop(req_id, None)
        else:
            self._offload_rejection_backoff[req_id] = (
                decision.reason,
                signature,
                step_id + MCP_KV_OFFLOAD_BACKOFF_STEPS,
            )
        return (decision.allowed, requested_preservation,
                decision.queue_fit_request_id)

    def _manage_mcp_kv_cache_offload(self):
        if not self.enable_kvcache_cpu_offloading:
            return

        pressure = self._get_pressure_snapshot()
        if pressure.waiting_request_count <= 0:
            self.mcp_manager.increment_coordination_counter(
                "offload_fast_gate_skip", reason="no-waiting-request")
            return

        requests_to_offload = self.mcp_manager.get_requests_to_offload()
        if len(requests_to_offload) > 0:
            logger.debug(
                "[scheduler] pending MCP offloads: count=%s",
                len(requests_to_offload),
            )

        eviction_plan = self._find_best_fit_waiting_request(
            MCP_KV_OFFLOAD_EVICTION_WINDOW_BLOCKS, pressure)
        fit_request, _, expected_evictions = eviction_plan
        if fit_request is None or expected_evictions <= 0:
            self.mcp_manager.increment_coordination_counter(
                "offload_fast_gate_skip", reason="no-runnable-eviction-demand")
            return

        h2d_target_ids = set(
            self.kv_cache_manager.block_pool.step_h2d_swap_map.values())
        computed_blocks, _, _ = self.kv_cache_manager.peek_computed_blocks(
            fit_request)
        waiting_hit_ids = {block.block_id for block in computed_blocks}
        imminent_eviction_ids = (
            self.kv_cache_manager.block_pool.get_imminent_eviction_block_ids(
                expected_evictions,
                excluded_block_ids=waiting_hit_ids,
            ))
        selected_gpu_source_ids = set(
            self.kv_cache_manager.block_pool.step_d2h_swap_map)
        req_blocks_to_offload = self._select_reusable_eviction_frontier(
            imminent_eviction_ids,
            selected_gpu_source_ids | h2d_target_ids,
            min(expected_evictions, MCP_KV_OFFLOAD_MAX_RELIEF_BLOCKS),
        )
        if not req_blocks_to_offload:
            self.mcp_manager.increment_coordination_counter(
                "offload_fast_gate_skip", reason="no-reusable-eviction-blocks")
            return

        for req_id in requests_to_offload:
            allowed, offload_count, admission_target = (
                self._plan_offload_request(req_id,
                                           req_blocks_to_offload,
                                           eviction_plan=eviction_plan))
            if not allowed or offload_count <= 0:
                continue

            offloaded_cpu_blocks: list[KVCacheBlock] = []
            copied_blocks = 0
            h2d_source_ids = set(
                self.kv_cache_manager.block_pool.step_h2d_swap_map)
            selected_blocks = req_blocks_to_offload[:offload_count]
            selected_shadow_ids: set[int] = set()
            shadow_misses = 0
            for block in selected_blocks:
                shadow = self.kv_cache_manager.block_pool.peek_cpu_shadow_block(
                    block.block_hash, h2d_source_ids)
                if shadow is None:
                    shadow_misses += 1
                elif shadow.ref_cnt == 0:
                    selected_shadow_ids.add(shadow.block_id)
            block_pool = self.kv_cache_manager.block_pool
            free_cpu_queue = block_pool.free_cpu_block_queue
            excluded_free_cpu_blocks = sum(
                1 for block_id in h2d_source_ids
                if free_cpu_queue.contains(block_pool.cpu_blocks[block_id]))
            safe_free_cpu_blocks = (block_pool.get_num_free_cpu_blocks() -
                                    excluded_free_cpu_blocks)
            if (len(selected_shadow_ids) + shadow_misses
                    > safe_free_cpu_blocks):
                self.mcp_manager.record_coordination_event(
                    "offload_commit_deferred",
                    req_id,
                    reason="insufficient-safe-cpu-blocks",
                    requested_blocks=offload_count,
                    safe_free_cpu_blocks=safe_free_cpu_blocks,
                )
                continue

            for block in selected_blocks:
                cpu_block, copied = (self.kv_cache_manager.block_pool.
                                     acquire_cpu_block_for_offload(
                                         block,
                                         excluded_cpu_block_ids=h2d_source_ids,
                                     ))
                if cpu_block is None:
                    break
                offloaded_cpu_blocks.append(cpu_block)
                copied_blocks += int(copied)
            if len(offloaded_cpu_blocks) < offload_count:
                for cpu_block in offloaded_cpu_blocks:
                    self.kv_cache_manager.block_pool.free_cpu_blocks(
                        [cpu_block])
                self.mcp_manager.record_coordination_event(
                    "offload_commit_deferred",
                    req_id,
                    reason="cpu-capacity-changed",
                    requested_blocks=offload_count,
                    acquired_blocks=len(offloaded_cpu_blocks),
                )
                continue
            self._ensure_agent_offload_coordinator().record_offload_committed(
                req_id)
            copied_gpu_block_ids = [
                block.block_id for block in selected_blocks if block.block_id
                in self.kv_cache_manager.block_pool.step_d2h_swap_map
            ]
            copied_cpu_block_ids = [
                self.kv_cache_manager.block_pool.step_d2h_swap_map[block_id]
                for block_id in copied_gpu_block_ids
            ]
            if copied_gpu_block_ids:
                self.mcp_manager.record_transfer_pending(
                    req_id,
                    direction="d2h",
                    gpu_blocks=copied_gpu_block_ids,
                    cpu_blocks=copied_cpu_block_ids,
                )
            else:
                self.mcp_manager.record_transfer_ready(req_id, "d2h")
            self.mcp_manager.record_coordination_event(
                "offload_committed",
                req_id,
                reason="offloaded",
                offloaded_blocks=len(offloaded_cpu_blocks),
                copied_blocks=copied_blocks,
                shadow_hit_blocks=len(offloaded_cpu_blocks) - copied_blocks,
                intended_admission_request_id=admission_target,
            )
            self.mcp_manager.mark_request_offloaded(req_id,
                                                    offloaded_cpu_blocks)
            pressure = self._get_pressure_snapshot()
            pressure.cpu_free_blocks = max(
                0, pressure.cpu_free_blocks - copied_blocks)
            break

    def _protect_unallocated_d2h_sources(self) -> tuple[int, int]:
        block_pool = self.kv_cache_manager.block_pool
        source_ids = tuple(block_pool.step_d2h_swap_map)
        free_sources = []
        reused_sources = 0
        for block_id in source_ids:
            block = block_pool.blocks[block_id]
            if (block.ref_cnt == 0
                    and block_pool.free_block_queue.contains(block)):
                free_sources.append(block)
            else:
                reused_sources += 1
        protected = block_pool.protect_d2h_gpu_blocks(free_sources)
        if source_ids:
            self.mcp_manager.record_coordination_event(
                "d2h_post_schedule_source_state",
                reason="capacity-neutral-source-protection",
                d2h_sources=len(source_ids),
                protected_sources=protected,
                reused_by_current_batch=reused_sources,
            )
        return protected, reused_sources

    def _update_req_agent_input_info(self, request: Request, input_len: int):
        self.agent_info_manager.update_req_agent_input_info(request, input_len)

    def _update_req_agent_output_info(self, request: Request, output_len: int):
        self.agent_info_manager.update_req_agent_output_info(
            request, output_len)


    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_block_ids: dict[str, list[int]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        if self.enable_agent_scheduling:
            self.agent_scheduler.begin_schedule_step(self.waiting,
                                                     self.running)

        if (self.enable_kvcache_cpu_offloading
                and self.mcp_manager.has_scheduler_work(
                    has_waiting=bool(self.waiting))):
            self._manage_mcp_kv_cache()

        d2h_swap_map = {}

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            if request.request_id in self.scheduled_req_ids:
                req_index += 1
                continue

            num_new_tokens = (request.num_tokens_with_spec -
                              request.num_computed_tokens)
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)
            assert num_new_tokens > 0

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - request.num_computed_tokens)
            num_new_tokens = self._cap_agent_prefill_tokens(
                request, num_new_tokens)
            assert num_new_tokens > 0

            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_budget)
                if num_new_tokens == 0:
                    req_index += 1
                    continue
            else:
                encoder_inputs_to_schedule = None
                new_encoder_budget = encoder_budget

            while True:
                agent_can_schedule = True
                if self.enable_agent_scheduling:
                    agent_can_schedule = self.agent_scheduler.can_schedule_req(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                new_blocks = None
                if agent_can_schedule:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY or self.policy == SchedulingPolicy.AGENT:
                        preempted_req = self._select_preemption_victim(request)
                        preempted_index = self.running.index(preempted_req)
                        self.running.remove(preempted_req)
                        if preempted_index < req_index:
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    if preempted_req in scheduled_running_reqs:
                        scheduled_running_reqs.remove(preempted_req)
                    rolled_back_tokens = num_scheduled_tokens.pop(
                        preempted_req.request_id, 0)
                    token_budget += rolled_back_tokens
                    req_to_new_block_ids.pop(preempted_req.request_id, None)
                    self.scheduled_req_ids.discard(preempted_req.request_id)
                    scheduled_spec_decode_tokens.pop(preempted_req.request_id,
                                                     None)
                    preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                        preempted_req.request_id, ())
                    released_encoder_budget = sum(
                        preempted_req.get_num_encoder_tokens(i)
                        for i in preempted_encoder_inputs)
                    encoder_budget += released_encoder_budget
                    new_encoder_budget += released_encoder_budget
                    self.encoder_cache_manager.free(preempted_req)

                    preempted_req.status = RequestStatus.FINISHED_PREEMPTED
                    if self.enable_agent_scheduling:
                        self.agent_scheduler.record_request_preempted(
                            preempted_req)
                    self.kv_cache_manager.free(preempted_req)
                    self.preempted_stop_requests.append(preempted_req)
                    preempted_reqs.append(preempted_req)
                    logger.info(
                        f"[scheduler] preempted_req: {preempted_req.request_id}, "
                        f"status: FINISHED_PREEMPTED, priority: "
                        f"{preempted_req.priority}, agent_info: "
                        f"{preempted_req.agent_info}")
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    if self.enable_agent_scheduling:
                        self.agent_scheduler.record_request_new_blocks(
                            request, new_blocks)
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            self.scheduled_req_ids.add(request.request_id)

            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in new_blocks
            ]
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = {
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            }
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(
            self.policy, self.agent_info_manager)

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()
                if request.status not in (RequestStatus.WAITING,
                                          RequestStatus.PREEMPTED,
                                          RequestStatus.WAITING_FOR_FSM):
                    logger.warning(
                        "[scheduler] drop stale waiting request %s with status %s",
                        request.request_id,
                        request.status,
                    )
                    self._remove_waiting_request(request)
                    continue
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self._remove_waiting_request(request)
                        skipped_waiting_requests.prepend_request(request)
                        continue

                if request.request_id in getattr(self,
                                                 "_waiting_h2d_prefetches",
                                                 {}):
                    self._remove_waiting_request(request)
                    skipped_waiting_requests.prepend_request(request)
                    continue

                if self.lora_config and request.lora_request and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id
                        not in scheduled_loras):
                    self._remove_waiting_request(request)
                    skipped_waiting_requests.prepend_request(request)
                    continue

                # Get already-cached tokens.
                computed_blocks, computed_cpu_blocks, num_computed_tokens = \
                    self.kv_cache_manager.get_computed_blocks(request)
                num_external_tokens = (
                    0 if self.connector is None else
                    self.connector.get_num_new_matched_tokens(
                        request, num_computed_tokens))
                num_computed_tokens += num_external_tokens

                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed requests,
                # which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                if (0 < self.scheduler_config.long_prefill_token_threshold <
                        num_new_tokens):
                    num_new_tokens = (
                        self.scheduler_config.long_prefill_token_threshold)
                num_new_tokens = min(num_new_tokens, token_budget)
                num_new_tokens = self._cap_agent_prefill_tokens(
                    request, num_new_tokens)
                assert num_new_tokens > 0

                agent_can_schedule = self.agent_scheduler.can_schedule_req(
                    request,
                    num_new_tokens + num_external_tokens,
                    new_computed_blocks=computed_blocks,
                    new_computed_cpu_blocks=computed_cpu_blocks,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )
                if not agent_can_schedule:
                    self.agent_scheduler.record_request_deferred(
                        request)
                    self._remove_waiting_request(request)
                    skipped_waiting_requests.prepend_request(request)
                    continue

                if request.has_encoder_inputs:
                    (encoder_inputs_to_schedule, num_new_tokens,
                     new_encoder_budget) = self._try_schedule_encoder_inputs(
                         request, num_computed_tokens, num_new_tokens,
                         encoder_budget)
                    if num_new_tokens == 0:
                        break
                else:
                    encoder_inputs_to_schedule = None
                    new_encoder_budget = encoder_budget

                self._update_req_agent_input_info(request,
                                                  request.num_prompt_tokens)

                step_h2d_swap_map = getattr(
                    self.kv_cache_manager.block_pool,
                    "step_h2d_swap_map",
                    {},
                )
                h2d_mapping_before = set(step_h2d_swap_map)
                allocation_options = {}
                if self.enable_kvcache_cpu_offloading:
                    allocation_options["excluded_h2d_gpu_block_ids"] = set(
                        self.kv_cache_manager.block_pool.step_d2h_swap_map)
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request=request,
                    num_tokens=num_new_tokens + num_external_tokens,
                    new_computed_blocks=computed_blocks,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                    new_computed_cpu_blocks=computed_cpu_blocks,
                    **allocation_options,
                )
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                request_cpu_block_ids = {
                    block.block_id
                    for block in computed_cpu_blocks
                }
                request_h2d_map = {
                    cpu_block_id: gpu_block_id
                    for cpu_block_id, gpu_block_id in
                    step_h2d_swap_map.items()
                    if cpu_block_id in request_cpu_block_ids
                    and cpu_block_id not in h2d_mapping_before
                }
                if request_h2d_map:
                    if not hasattr(self, "_waiting_h2d_prefetches"):
                        self._waiting_h2d_prefetches = {}
                    self._waiting_h2d_prefetches[request.request_id] = {
                        "gpu_blocks": set(request_h2d_map.values()),
                        "cpu_blocks": set(request_h2d_map),
                    }
                    self.mcp_manager.record_transfer_pending(
                        request.request_id,
                        direction="h2d",
                        gpu_blocks=list(request_h2d_map.values()),
                        cpu_blocks=list(request_h2d_map),
                    )
                    self.mcp_manager.record_coordination_event(
                        "waiting_h2d_prefetch",
                        request.request_id,
                        reason="cpu-prefix-hit",
                        logical_blocks=len(request_h2d_map),
                    )
                    if self.enable_agent_scheduling:
                        self.agent_scheduler.record_request_new_blocks(
                            request, new_blocks)
                    self._remove_waiting_request(request)
                    skipped_waiting_requests.prepend_request(request)
                    continue

                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request, num_external_tokens)

                if self.enable_agent_scheduling:
                    self.agent_scheduler.record_request_new_blocks(
                        request, new_blocks)

                self._remove_waiting_request(request)

                req_index += 1
                self.running.append(request)
                self.scheduled_req_ids.add(request.request_id)

                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in (computed_blocks + new_blocks)
                ]
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens

                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        scheduled_request_ids = {
            request.request_id
            for requests in (scheduled_new_reqs, scheduled_resumed_reqs,
                             scheduled_running_reqs)
            for request in requests
        }
        assert set(num_scheduled_tokens) == scheduled_request_ids
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (
            len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
            len(scheduled_running_reqs) <= len(self.running)
        ), f"scheduled_new_reqs: {len(scheduled_new_reqs)}, scheduled_resumed_reqs: {len(scheduled_resumed_reqs)}, scheduled_running_reqs: {len(scheduled_running_reqs)}, running: {len(self.running)}"

        if (self.enable_kvcache_cpu_offloading
                and self.kv_cache_manager.block_pool.step_d2h_swap_map):
            self._protect_unallocated_d2h_sources()
            d2h_swap_map = self.kv_cache_manager.get_d2h_swap_map()
            self.kv_cache_manager.clear_step_d2h_swap_map()

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = 0
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(
                any_request, len(self.running))

        structured_output_request_ids = {
            request.request_id: index
            for index, request in enumerate(self.running)
            if request.use_structured_output
            and request.request_id in num_scheduled_tokens
        }
        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests, structured_output_request_ids, len(self.running))

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=True,
            ) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_reqs
        ]

        if (self.enable_kvcache_cpu_offloading
                and self.kv_cache_manager.block_pool.step_h2d_swap_map):
            h2d_swap_map = self.kv_cache_manager.get_h2d_swap_map()
            self.kv_cache_manager.clear_step_h2d_swap_map()
        else:
            h2d_swap_map = {}

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
            d2h_swap_map=d2h_swap_map,
            h2d_swap_map=h2d_swap_map,
        )

        if self.connector is not None:
            scheduler_output.kv_connector_metadata = (
                self.connector.build_connector_meta(scheduler_output))

        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += num_scheduled_token

        self.finished_req_ids = set()
        return scheduler_output

    def _make_cached_request_data(
        self,
        request: Request,
        num_scheduled_tokens: int,
        num_scheduled_spec_tokens: int,
        new_block_ids: list[int],
        resumed_from_preemption: bool,
    ) -> CachedRequestData:
        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        num_computed_tokens = request.num_computed_tokens
        num_regular_tokens = num_scheduled_tokens - num_scheduled_spec_tokens
        new_token_ids = request.all_token_ids[
            num_computed_tokens:num_computed_tokens + num_regular_tokens]
        req_data = self._cached_reqs_data.get(request.request_id)
        if req_data is not None:
            req_data.resumed_from_preemption = resumed_from_preemption
            req_data.new_token_ids = new_token_ids
            req_data.new_block_ids = new_block_ids
            req_data.num_computed_tokens = num_computed_tokens
        else:
            req_data = CachedRequestData.from_request(request,
                                                      resumed_from_preemption,
                                                      new_token_ids,
                                                      new_block_ids)
            self._cached_reqs_data[request.request_id] = req_data
        return req_data

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_budget: int,
    ) -> tuple[list[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        encoder_inputs_to_schedule: list[int] = []
        mm_positions = request.mm_positions
        assert mm_positions is not None
        assert len(mm_positions) > 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info.offset
            num_encoder_tokens = pos_info.length

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break
            if start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if self.encoder_cache_manager.has_cache(request, i):
                # The encoder input is already computed and cached.
                continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (self.scheduler_config.disable_chunked_mm_input
                    and num_computed_tokens < start_pos
                    and (num_computed_tokens + num_new_tokens)
                    < (start_pos + num_encoder_tokens)):
                num_new_tokens = start_pos - num_computed_tokens
                break

            if (not self.encoder_cache_manager.can_allocate(request, i)
                    or num_encoder_tokens > encoder_budget):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            encoder_budget -= num_encoder_tokens
            encoder_inputs_to_schedule.append(i)
        return encoder_inputs_to_schedule, num_new_tokens, encoder_budget

    def _apply_cpu_offload_completion_events(
            self, completion_events: list[dict]) -> list[str]:
        if not completion_events:
            return []
        kv_cache_manager = getattr(self, "kv_cache_manager", None)
        block_pool = getattr(kv_cache_manager, "block_pool", None)
        release_d2h = getattr(block_pool, "release_d2h_gpu_blocks", None)
        publish_d2h = getattr(block_pool, "publish_completed_d2h_gpu_blocks",
                              None)
        if release_d2h is not None:
            for event in completion_events:
                if event.get("direction") == "d2h":
                    gpu_blocks = event.get("gpu_blocks", ())
                    release_d2h(gpu_blocks)
                    if publish_d2h is not None and not event.get("failed"):
                        publish_d2h(gpu_blocks)
        reactivated_requests = self.mcp_manager.record_transfer_completions(
            completion_events)
        if reactivated_requests:
            for request_id in reactivated_requests:
                self._offload_rejection_backoff.pop(request_id, None)
            self.mcp_manager.record_coordination_event(
                "transfer_completion_reactivation",
                reason="cuda-event-complete",
                completed_transfers=len(completion_events),
                reactivated_requests=reactivated_requests,
            )
        waiting_prefetches = getattr(self, "_waiting_h2d_prefetches", {})
        for request_id, transfer in list(waiting_prefetches.items()):
            completed = any(
                event.get("direction") == "h2d" and not event.get("failed")
                and transfer["gpu_blocks"].issubset(
                    set(event.get("gpu_blocks",
                                  ()))) and transfer["cpu_blocks"].issubset(
                                      set(event.get("cpu_blocks", ())))
                for event in completion_events)
            if completed:
                waiting_prefetches.pop(request_id, None)
                self._offload_rejection_backoff.pop(request_id, None)
                if request_id not in reactivated_requests:
                    reactivated_requests.append(request_id)
        return reactivated_requests

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:
        completion_events = getattr(model_runner_output,
                                    "cpu_offload_completion_events", [])
        self._apply_cpu_offload_completion_events(completion_events)

        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_token_ids = model_runner_output.spec_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        new_running: list[Request] = []
        outputs: list[EngineCoreOutput] = []
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        for request in self.preempted_stop_requests:
            self._free_request(request)
            if request.request_id in self.scheduled_req_ids:
                self.scheduled_req_ids.remove(request.request_id)
            outputs.append(
                EngineCoreOutput(request_id=request.request_id,
                                 new_token_ids=[],
                                 finish_reason=request.get_finished_reason(),
                                 new_logprobs=None,
                                 new_prompt_logprobs_tensors=None,
                                 stop_reason=request.stop_reason,
                                 events=request.take_events()))
        self.preempted_stop_requests.clear()

        # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        for request in self.running:
            req_id = request.request_id
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                new_running.append(request)
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index]

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens, where is given by:
                # len(scheduled_spec_token_ids) + 1 - len(generated_token_ids).
                num_tokens_rejected = (len(scheduled_spec_token_ids) + 1 -
                                       len(generated_token_ids))
                request.num_computed_tokens -= num_tokens_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=len(scheduled_spec_token_ids),
                    num_accepted_tokens=len(generated_token_ids) - 1)

            cached_encoder_input_ids = (
                self.encoder_cache_manager.get_cached_input_ids(request))
            # OPTIMIZATION: Avoid list(set) if the set is empty.
            if cached_encoder_input_ids:
                for input_id in list(cached_encoder_input_ids):
                    mm_positions = request.mm_positions[input_id]
                    start_pos = mm_positions.offset
                    num_tokens = mm_positions.length
                    if start_pos + num_tokens <= request.num_computed_tokens:
                        # The encoder output is already processed and stored
                        # in the decoder's KV cache.
                        self.encoder_cache_manager.free_encoder_input(
                            request, input_id)

            # Add newly generated spec token ids to the request.
            if spec_token_ids is not None:
                request.spec_token_ids = spec_token_ids[req_index]

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids

            # Append generated tokens and check for stop. Note that if
            # a request is still being prefilled, we expect the model runner
            # to return empty token ids for the request.
            for num_new, output_token_id in enumerate(new_token_ids, 1):
                request.append_output_token_ids(output_token_id)

                # Check for stop and update request state.
                # This must be called before we make the EngineCoreOutput.
                stopped = check_stop(request, self.max_model_len)
                if stopped:
                    self._free_request(request)
                    del new_token_ids[num_new:]  # Trim new tokens if needed.
                    break

            # Extract sample logprobs if needed.
            if request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and request.use_structured_output:
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids:
                # Add EngineCoreOutput for this Request.
                outputs.append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason,
                        events=request.take_events()))
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

            self.scheduled_req_ids.remove(req_id)
            if not stopped:
                new_running.append(request)

        self.running = new_running
        engine_core_outputs = EngineCoreOutputs(
            outputs=outputs,
            scheduler_stats=self.make_stats(spec_decoding_stats),
        )
        if self.include_finished_set:
            #TODO currently sending duplicates here, improve this
            engine_core_outputs.finished_requests = (
                scheduler_output.finished_req_ids | self.finished_req_ids)

        return engine_core_outputs

    def add_request(self, request: Request) -> None:
        self.mcp_manager.register_request(
            request.request_id,
            prediction_key=request.agent_type or request.agent_name
            or request.request_id,
            metadata=request.agent_info,
        )
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = []
        waiting_requests_to_remove = []
        valid_requests = []

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.append(request)
                self.scheduled_req_ids.discard(request.request_id)
            elif request.status == RequestStatus.FINISHED_PREEMPTED:
                self.preempted_stop_requests.remove(request)
            else:
                waiting_requests_to_remove.append(request)

        for request in running_requests_to_remove:
            self.running.remove(request)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> None:
        assert request.is_finished()

        request_id = request.request_id
        getattr(self, "_waiting_h2d_prefetches", {}).pop(request_id, None)
        self.mcp_manager.cancel_request_transfers(request_id)

        blocks = self.kv_cache_manager.req_to_blocks.get(request_id, [])
        if self.enable_kvcache_cpu_offloading:
            logger.debug(
                "[scheduler] save finished request %s blocks: %s",
                request_id,
                len(blocks),
            )
            self.mcp_manager.save_finished_request_blocks(request_id, blocks)

        offloaded_cpu_blocks = self.mcp_manager.get_offloaded_request_data(
            request_id)
        if offloaded_cpu_blocks:
            self.kv_cache_manager.block_pool.free_cpu_blocks(
                offloaded_cpu_blocks)
            self.mcp_manager.mark_request_uploaded(request_id)
        self._update_req_agent_output_info(request, request.num_output_tokens)

        if self.enable_agent_scheduling:
            self.agent_scheduler.finish_request(request, blocks)

        self.kv_cache_manager.free(request)
        self.kv_cache_manager.free_block_hashes(request)
        self.encoder_cache_manager.free(request)
        self._cached_reqs_data.pop(request.request_id, None)
        del self.requests[request.request_id]
        self.finished_req_ids.add(request.request_id)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def get_num_unscheduled_requests(self) -> int:
        """Number of requests that are not being processed by the executor."""
        return self.get_num_unfinished_requests() - len(self.scheduled_req_ids)

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None

        cpu_cache_usage = self.kv_cache_manager.cpu_usage \
            if self.enable_kvcache_cpu_offloading else None
        cpu_prefix_cache_stats = self.kv_cache_manager.make_cpu_prefix_cache_stats() \
            if self.enable_kvcache_cpu_offloading else None
        swap_out_count = self.kv_cache_manager.get_swap_out_count() \
            if self.enable_kvcache_cpu_offloading else None
        swap_in_count = self.kv_cache_manager.get_swap_in_count() \
            if self.enable_kvcache_cpu_offloading else None

        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            gpu_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            cpu_cache_usage=cpu_cache_usage,
            cpu_prefix_cache_stats=cpu_prefix_cache_stats,
            swap_out_count=swap_out_count,
            swap_in_count=swap_in_count,
            gpu_evict_count=self.kv_cache_manager.get_gpu_cpu_evict_count()[0],
            cpu_evict_count=self.kv_cache_manager.get_gpu_cpu_evict_count()[1],
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats],
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> Optional[SpecDecodingStats]:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats()
        spec_decoding_stats.observe(num_draft_tokens=num_draft_tokens,
                                    num_accepted_tokens=num_accepted_tokens)
        return spec_decoding_stats
