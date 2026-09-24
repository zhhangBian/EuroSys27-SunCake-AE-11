# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector import (
    align_to_block_size, )
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class MooncakeReqMeta:
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool

    @staticmethod
    def make_meta(token_ids: list[int],
                  block_ids: list[int],
                  block_size: int,
                  is_store: bool,
                  num_tokens: Optional[int] = None) -> "MooncakeReqMeta":
        valid_num_tokens = (align_to_block_size(len(token_ids), block_size)
                            if num_tokens is None else num_tokens)
        if valid_num_tokens < 0 or valid_num_tokens % block_size != 0:
            raise ValueError("Mooncake prefix length must be block aligned")
        if valid_num_tokens > len(token_ids):
            raise ValueError("Mooncake prefix length exceeds token count")
        if len(block_ids) * block_size < valid_num_tokens:
            raise ValueError(
                "Mooncake prefix metadata has fewer KV slots than tokens: "
                f"tokens={valid_num_tokens} blocks={len(block_ids)}")
        token_ids_tensor = torch.tensor(token_ids,
                                        dtype=torch.int64)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids, dtype=torch.int64)
        block_offsets = torch.arange(0, block_size, dtype=torch.int64)
        slot_mapping = block_offsets.reshape(
            (1, block_size)) + (block_ids_tensor.reshape((-1, 1)) * block_size)
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return MooncakeReqMeta(token_ids=token_ids_tensor,
                               slot_mapping=slot_mapping,
                               is_store=is_store)


@dataclass
class MooncakeStoreConnectorMetadata(KVConnectorMetadata):
    requests: list[MooncakeReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(self,
                    token_ids: list[int],
                    block_ids: list[int],
                    block_size: int,
                    is_store: bool,
                    num_tokens: Optional[int] = None) -> None:
        self.requests.append(
            MooncakeReqMeta.make_meta(token_ids, block_ids, block_size,
                                      is_store, num_tokens))


class MooncakeStoreConnectorV1(KVConnectorBase_V1):
    """V1 Mooncake connector for exact-prefix remote KV reuse.

    This mirrors the debug SharedStorageConnector control flow but uses
    MooncakeStore as the backing remote CPU memory pool.
    """

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, tuple[Request, int]] = {}
        self._matched_token_counts: dict[str, int] = {}
        self._request_prompt_token_ids: dict[str, list[int]] = {}
        self._request_block_ids: dict[str, list[int]] = {}
        self._stored_request_ids: set[str] = set()
        self._tp_rank = 0
        self._kv_store = None
        self._pending_marker_keys: set[str] = set()
        from vllm.distributed.kv_transfer.kv_lookup_buffer.mooncake_store import (
            MooncakeStore)
        self._kv_store = MooncakeStore(vllm_config)
        if role is KVConnectorRole.WORKER:
            from vllm.distributed.parallel_state import get_world_group

            self._tp_rank = get_world_group().local_rank
        logger.info("Initialized MooncakeStoreConnectorV1 role=%s tp_rank=%s",
                    role, self._tp_rank)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        metadata = self._get_connector_metadata()
        logger.info(
            "[MooncakeConnectorStats] %s",
            json.dumps(
                {
                    "op": "v1_start_load",
                    "metadata_type": type(metadata).__name__,
                },
                sort_keys=True))
        if metadata is None:
            return
        assert isinstance(metadata, MooncakeStoreConnectorMetadata)
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        for request in metadata.requests:
            if request.is_store:
                continue
            start_time = time.perf_counter()
            loaded_layers = 0
            for layer_name in forward_context.no_compile_layers:
                attn_layer = forward_context.no_compile_layers[layer_name]
                kv_cache_layer = attn_layer.kv_cache[
                    forward_context.virtual_engine]
                key = self._make_layer_key(layer_name, request.token_ids)
                remote_kv = self._kv_store.get(key) if self._kv_store else None
                if remote_kv is None:
                    logger.info(
                        "[MooncakeConnectorStats] %s",
                        json.dumps(
                            {
                                "op": "v1_load_layer",
                                "hit": False,
                                "layer": layer_name,
                                "tokens": int(request.token_ids.shape[0]),
                                "tp_rank": int(self._tp_rank),
                            },
                            sort_keys=True),
                    )
                    continue
                self._inject_kv_into_layer(kv_cache_layer, remote_kv,
                                           request.slot_mapping, attn_metadata)
                loaded_layers += 1
                logger.info(
                    "[MooncakeConnectorStats] %s",
                    json.dumps(
                        {
                            "op": "v1_load_layer",
                            "hit": True,
                            "layer": layer_name,
                            "tokens": int(request.token_ids.shape[0]),
                            "tp_rank": int(self._tp_rank),
                        },
                        sort_keys=True),
                )
            logger.info(
                "[MooncakeConnectorStats] %s",
                json.dumps(
                    {
                        "op": "v1_load_request",
                        "tokens": int(request.token_ids.shape[0]),
                        "loaded_layers": loaded_layers,
                        "requested_layers": len(
                            forward_context.no_compile_layers),
                        "elapsed_ms":
                        (time.perf_counter() - start_time) * 1000.0,
                        "tp_rank": int(self._tp_rank),
                    },
                    sort_keys=True),
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        if metadata is None:
            return
        assert isinstance(metadata, MooncakeStoreConnectorMetadata)
        for request in metadata.requests:
            if not request.is_store:
                continue
            start_time = time.perf_counter()
            kv_tensor = self._extract_kv_from_layer(kv_layer,
                                                    request.slot_mapping,
                                                    attn_metadata)
            key = self._make_layer_key(layer_name, request.token_ids)
            if self._kv_store is not None:
                self._kv_store.put(key, kv_tensor.detach())
            self._pending_marker_keys.add(
                self._make_marker_key(request.token_ids))
            logger.info(
                "[MooncakeConnectorStats] %s",
                json.dumps(
                    {
                        "op": "v1_save_layer",
                        "layer": layer_name,
                        "tokens": int(request.token_ids.shape[0]),
                        "elapsed_ms":
                        (time.perf_counter() - start_time) * 1000.0,
                        "tp_rank": int(self._tp_rank),
                    },
                    sort_keys=True),
            )

    def wait_for_save(self):
        if self._kv_store is not None:
            for marker_key in sorted(self._pending_marker_keys):
                marker = torch.tensor([1], dtype=torch.int8)
                self._kv_store.put(marker_key, marker)
        self._pending_marker_keys.clear()
        return None

    def get_num_new_matched_tokens(self, request: "Request",
                                   num_computed_tokens: int) -> int:
        matched_token_count = self._get_matched_token_count(request)
        self._matched_token_counts[request.request_id] = matched_token_count
        if matched_token_count <= 0:
            logger.info(
                "[MooncakeConnectorStats] %s",
                json.dumps(
                    {
                        "op": "v1_scheduler_match",
                        "hit": False,
                        "matched_tokens": 0,
                        "request_id": request.request_id,
                    },
                    sort_keys=True))
            return 0
        matched = max(0, matched_token_count - num_computed_tokens)
        logger.info(
            "[MooncakeConnectorStats] %s",
            json.dumps(
                {
                    "op": "v1_scheduler_match",
                    "hit": matched > 0,
                    "matched_tokens": matched,
                    "request_id": request.request_id,
                },
                sort_keys=True))
        return matched

    def update_state_after_alloc(self, request: "Request",
                                 num_external_tokens: int):
        if num_external_tokens > 0:
            matched_token_count = self._matched_token_counts.pop(
                request.request_id,
                request.num_computed_tokens + num_external_tokens)
            self._requests_need_load[request.request_id] = (
                request, matched_token_count)
        else:
            self._matched_token_counts.pop(request.request_id, None)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = MooncakeStoreConnectorMetadata()
        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._requests_need_load:
                request, matched_tokens = self._requests_need_load[
                    new_req.req_id]
                meta.add_request(
                    token_ids=request.prompt_token_ids,
                    block_ids=new_req.block_ids,
                    block_size=self._block_size,
                    is_store=False,
                    num_tokens=matched_tokens,
                )
                total_need_load += 1
            elif not self._found_match_for_request(new_req):
                self._request_prompt_token_ids[new_req.req_id] = list(
                    new_req.prompt_token_ids)
                self._request_block_ids[new_req.req_id] = list(
                    new_req.block_ids)
                computed_after = (
                    new_req.num_computed_tokens +
                    scheduler_output.num_scheduled_tokens[new_req.req_id])
                self._maybe_add_store_request(meta, new_req.req_id,
                                              computed_after)

        for cached_req in scheduler_output.scheduled_cached_reqs:
            if cached_req.req_id in self._requests_need_load:
                request, matched_tokens = self._requests_need_load[
                    cached_req.req_id]
                meta.add_request(
                    token_ids=request.prompt_token_ids,
                    block_ids=cached_req.new_block_ids,
                    block_size=self._block_size,
                    is_store=False,
                    num_tokens=matched_tokens,
                )
                total_need_load += 1
                continue
            if cached_req.req_id not in self._request_prompt_token_ids:
                continue
            if cached_req.resumed_from_preemption:
                self._request_block_ids[cached_req.req_id] = list(
                    cached_req.new_block_ids)
            else:
                self._request_block_ids[cached_req.req_id].extend(
                    cached_req.new_block_ids)
            computed_after = (cached_req.num_computed_tokens +
                              len(cached_req.new_token_ids))
            self._maybe_add_store_request(meta, cached_req.req_id,
                                          computed_after)

        if total_need_load != len(self._requests_need_load):
            logger.warning(
                "Mooncake V1 load metadata mismatch: need=%s meta=%s",
                len(self._requests_need_load), total_need_load)
        self._requests_need_load.clear()
        for req_id in scheduler_output.finished_req_ids:
            self._matched_token_counts.pop(req_id, None)
            self._request_prompt_token_ids.pop(req_id, None)
            self._request_block_ids.pop(req_id, None)
            self._stored_request_ids.discard(req_id)
        logger.info(
            "[MooncakeConnectorStats] %s",
            json.dumps(
                {
                    "op": "v1_build_meta",
                    "requests": len(meta.requests),
                    "stores": sum(1 for req in meta.requests if req.is_store),
                    "loads": sum(1
                                 for req in meta.requests if not req.is_store),
                },
                sort_keys=True))
        return meta

    def _maybe_add_store_request(self, meta: MooncakeStoreConnectorMetadata,
                                 request_id: str, computed_after: int) -> None:
        if request_id in self._stored_request_ids:
            return
        prompt_token_ids = self._request_prompt_token_ids[request_id]
        if computed_after < len(prompt_token_ids):
            return
        meta.add_request(
            token_ids=prompt_token_ids,
            block_ids=self._request_block_ids[request_id],
            block_size=self._block_size,
            is_store=True,
        )
        self._stored_request_ids.add(request_id)
        self._request_prompt_token_ids.pop(request_id, None)
        self._request_block_ids.pop(request_id, None)

    def close(self) -> None:
        if self._kv_store is not None:
            self._kv_store.close()

    @staticmethod
    def _tensor_hash(tensor: torch.Tensor) -> str:
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()
        return hashlib.blake2b(tensor_bytes).hexdigest()[:16]

    def _make_layer_key(self, layer_name: str, token_ids: torch.Tensor) -> str:
        return f"v1_{self._tensor_hash(token_ids)}_{layer_name}_{self._tp_rank}"

    def _found_match_for_request(self, request: "Request") -> bool:
        return self._get_matched_token_count(request) > 0

    def _get_matched_token_count(self, request: "Request") -> int:
        num_tokens_to_check = align_to_block_size(
            len(request.prompt_token_ids), self._block_size)
        if num_tokens_to_check <= 0:
            return 0
        token_ids = torch.tensor(request.prompt_token_ids,
                                 dtype=torch.int64)[:num_tokens_to_check]
        # Store/load metadata is exact-prefix based. We use a marker key for
        # cheap scheduler-side match checks without fetching every layer.
        marker_key = f"v1_marker_{self._tensor_hash(token_ids)}_{self._tp_rank}"
        if self._kv_store is None:
            return 0
        return num_tokens_to_check if self._kv_store.get(
            marker_key) is not None else 0

    def _make_marker_key(self, token_ids: torch.Tensor) -> str:
        return f"v1_marker_{self._tensor_hash(token_ids)}_{self._tp_rank}"

    @staticmethod
    def _inject_kv_into_layer(dst_kv_cache_layer: torch.Tensor,
                              src_kv_cache: torch.Tensor,
                              slot_mapping: torch.Tensor,
                              attn_metadata: "AttentionMetadata") -> None:
        token_axis = 0 if isinstance(attn_metadata, MLACommonMetadata) else 1
        if src_kv_cache.shape[token_axis] != slot_mapping.numel():
            raise RuntimeError(
                "Mooncake KV token count does not match destination slots: "
                f"source={src_kv_cache.shape[token_axis]} "
                f"slots={slot_mapping.numel()}")
        slot_mapping = slot_mapping.to(dst_kv_cache_layer.device,
                                       non_blocking=True)
        shape = dst_kv_cache_layer.shape
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages = shape[0]
            page_size = shape[1]
            dst = dst_kv_cache_layer.reshape(num_pages * page_size, -1)
            dst[slot_mapping, ...] = src_kv_cache.to(dst.device,
                                                     non_blocking=True)
            dst.reshape(shape)
            return
        num_pages = shape[1]
        page_size = shape[2]
        dst = dst_kv_cache_layer.reshape(2, num_pages * page_size, -1)
        dst[:, slot_mapping, ...] = src_kv_cache.to(dst.device,
                                                    non_blocking=True)
        dst.reshape(shape)

    @staticmethod
    def _extract_kv_from_layer(
            layer: torch.Tensor, slot_mapping: torch.Tensor,
            attn_metadata: "AttentionMetadata") -> torch.Tensor:
        slot_mapping = slot_mapping.to(layer.device, non_blocking=True)
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = layer.shape[0], layer.shape[1]
            return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
        num_pages, page_size = layer.shape[1], layer.shape[2]
        return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping,
                                                           ...]
