# SPDX-License-Identifier: Apache-2.0
import os
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock, BlockHashType, FreeBlockQueueWithBuffer

logger = init_logger(__name__)


@dataclass
class _PendingPreservation:
    block_hash: BlockHashType
    cpu_block_id: Optional[int]
    gpu_block_id: Optional[int] = None


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


class CpuOffloadingBlockPool(BlockPool):

    def __init__(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        enable_caching: bool = True,
    ):
        logger.info(
            "enable cpu offloading block pool, num gpu blocks: %s, num cpu blocks: %s",
            num_gpu_blocks, num_cpu_blocks)

        if num_cpu_blocks <= 0:
            raise ValueError("CPU offloading requires at least one CPU block")

        super().__init__(num_gpu_blocks, enable_caching)

        self.num_cpu_blocks = num_cpu_blocks
        self.cpu_blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_cpu_blocks)
        ]
        self.free_cpu_block_queue = FreeBlockQueueWithBuffer(self.cpu_blocks)

        self.cached_block_hash_to_cpu_block: \
            dict[BlockHashType, dict[int, KVCacheBlock]] \
            = defaultdict(dict)

        self.step_d2h_swap_map: dict[int, int] = {}
        self.step_h2d_swap_map: dict[int, int] = {}

        self.swap_in_count: int = 0
        self.swap_out_count: int = 0

        self.no_copy_restore_count: int = 0
        self.cpu_shadow_hit_count: int = 0
        self.d2h_logical_blocks: int = 0
        self.h2d_logical_blocks: int = 0
        self._next_preservation_id = 0
        self._pending_preservations: dict[int, _PendingPreservation] = {}
        self._cpu_preservation_ids: dict[int, int] = {}
        self._gpu_preservation_ids: dict[int, int] = {}
        self.preserved_d2h_blocks: int = 0
        self.proactive_gpu_published_blocks: int = 0
        self.effective_reuse_blocks: int = 0
        self.effective_cpu_reuse_blocks: int = 0
        self.effective_gpu_reuse_blocks: int = 0
        self.ineffective_reuse_blocks: int = 0
        self.prefix_lookup_hits_by_hash: dict[BlockHashType,
                                              int] = defaultdict(int)
        self.prefix_parent_by_hash: dict[BlockHashType,
                                         Optional[BlockHashType]] = {}
        self.prefix_depth_by_hash: dict[BlockHashType, int] = {}
        self.ineffective_preservations_by_hash: dict[BlockHashType,
                                                     int] = defaultdict(int)
        self.real_prefix_lookup_hit_blocks: int = 0

        self.cpu_evict_count: int = 0
        self._d2h_protected_gpu_blocks: dict[int,
                                             tuple[KVCacheBlock, Optional[int],
                                                   Optional[int]]] = {}
        default_shadow_blocks = max(1, min(num_cpu_blocks // 2,
                                           num_gpu_blocks))
        self.max_cpu_shadow_blocks = max(
            0,
            min(
                num_cpu_blocks,
                _env_int("VLLM_CPU_OFFLOAD_SHADOW_BLOCKS",
                         default_shadow_blocks),
            ),
        )
        self._cpu_shadow_lru: OrderedDict[int, None] = OrderedDict()

    def _remove_cached_cpu_block_mapping(
        self,
        cpu_block: KVCacheBlock,
        block_hash: Optional[BlockHashType] = None,
    ) -> None:
        block_hash = block_hash or cpu_block.block_hash
        if not block_hash:
            return

        cached_cpu_blocks = self.cached_block_hash_to_cpu_block.get(block_hash)
        if not cached_cpu_blocks:
            return

        cached_cpu_blocks.pop(cpu_block.block_id, None)
        if not cached_cpu_blocks:
            del self.cached_block_hash_to_cpu_block[block_hash]

    def touch_cpu_blocks(self, cpu_blocks: list[KVCacheBlock]) -> None:
        for cpu_block in cpu_blocks:
            if cpu_block.ref_cnt == 0:
                self.free_cpu_block_queue.remove(cpu_block)
            self._touch_cpu_shadow(cpu_block)
            cpu_block.incr_ref()

    def _touch_cpu_shadow(self, cpu_block: KVCacheBlock) -> None:
        if cpu_block.block_hash is None:
            return
        self._cpu_shadow_lru.pop(cpu_block.block_id, None)
        self._cpu_shadow_lru[cpu_block.block_id] = None

    def _drop_cpu_shadow(self, cpu_block: KVCacheBlock) -> None:
        self._cpu_shadow_lru.pop(cpu_block.block_id, None)

    def _register_cpu_shadow(self, cpu_block: KVCacheBlock) -> None:
        if cpu_block.block_hash is None:
            return
        if self.max_cpu_shadow_blocks <= 0:
            self._maybe_evict_cached_cpu_block(cpu_block)
            return
        self._touch_cpu_shadow(cpu_block)
        self._evict_cpu_shadow_overflow()

    def _evict_cpu_shadow_overflow(self) -> None:
        while len(self._cpu_shadow_lru) > self.max_cpu_shadow_blocks:
            evicted = False
            for block_id in tuple(self._cpu_shadow_lru):
                cpu_block = self.cpu_blocks[block_id]
                if cpu_block.ref_cnt != 0:
                    continue
                self._maybe_evict_cached_cpu_block(cpu_block)
                evicted = True
                break
            if not evicted:
                break

    def _bind_gpu_and_cpu_block(
        self,
        gpu_block: KVCacheBlock,
        cpu_block: KVCacheBlock,
        *,
        keep_gpu_cached: bool,
    ) -> None:
        gpu_block.block_in_cpu = True
        gpu_block.related_block = cpu_block
        cpu_block.related_block = gpu_block
        if not keep_gpu_cached and gpu_block.block_hash is not None:
            self._remove_cached_block_mapping(gpu_block, gpu_block.block_hash)

    def offload_block(self, gpu_block: KVCacheBlock,
                      cpu_block: KVCacheBlock) -> None:
        self.add_d2h_swap_map(gpu_block.block_id, cpu_block.block_id)
        self._bind_gpu_and_cpu_block(gpu_block,
                                     cpu_block,
                                     keep_gpu_cached=False)
        cpu_block.incr_ref()

        if gpu_block.block_hash is None:
            return

        cpu_block.block_hash = gpu_block.block_hash
        self.cached_block_hash_to_cpu_block[gpu_block.block_hash][
            cpu_block.block_id] = cpu_block
        self._start_preservation(cpu_block)

    def acquire_cpu_block_for_offload(
        self,
        gpu_block: KVCacheBlock,
        *,
        excluded_cpu_block_ids: Optional[set[int]] = None,
    ) -> tuple[Optional[KVCacheBlock], bool]:
        block_hash = gpu_block.block_hash
        if block_hash is None:
            return None, False

        excluded_cpu_block_ids = excluded_cpu_block_ids or set()
        related_cpu_block = gpu_block.related_block
        shadow = None
        if (related_cpu_block is not None
                and related_cpu_block.block_hash == block_hash
                and related_cpu_block.block_id not in excluded_cpu_block_ids):
            shadow = related_cpu_block
        else:
            cached_cpu_blocks = self.cached_block_hash_to_cpu_block.get(
                block_hash, {})
            for candidate in cached_cpu_blocks.values():
                if candidate.block_id not in excluded_cpu_block_ids:
                    shadow = candidate
                    break

        if shadow is not None:
            if shadow.ref_cnt == 0:
                self.free_cpu_block_queue.remove(shadow)
            shadow.incr_ref()
            self._touch_cpu_shadow(shadow)
            self._remove_cached_block_mapping(gpu_block, block_hash)
            gpu_block.block_in_cpu = True
            gpu_block.related_block = shadow
            published_source = shadow.related_block
            source_is_current = (published_source is not None
                                 and published_source.related_block is shadow
                                 and published_source.block_hash == block_hash)
            if not source_is_current:
                shadow.related_block = gpu_block
            self.cpu_shadow_hit_count += 1
            return shadow, False

        free_blocks_to_scan = self.get_num_free_cpu_blocks()
        skipped: list[KVCacheBlock] = []
        cpu_block: Optional[KVCacheBlock] = None
        for _ in range(free_blocks_to_scan):
            candidate = self.free_cpu_block_queue.popleft()
            if candidate.block_id in excluded_cpu_block_ids:
                skipped.append(candidate)
                continue
            cpu_block = candidate
            break
        for candidate in skipped:
            self.free_cpu_block_queue.append(candidate)
        if cpu_block is None:
            return None, False

        self._maybe_evict_cached_cpu_block(cpu_block)
        self.offload_block(gpu_block, cpu_block)
        return cpu_block, True

    def upload_block(
        self,
        cpu_block: KVCacheBlock,
        gpu_block: Optional[KVCacheBlock] = None,
    ) -> Optional[KVCacheBlock]:
        proactive_upload = gpu_block is not None
        block_hash = cpu_block.block_hash
        if block_hash is None:
            stale_gpu_block = cpu_block.related_block
            if stale_gpu_block is not None and stale_gpu_block.related_block is cpu_block:
                stale_gpu_block.block_in_cpu = False
                stale_gpu_block.related_block = None
            cpu_block.related_block = None
            cpu_block.reset_hash()
            return stale_gpu_block

        stale_gpu_block = cpu_block.related_block
        reusable_gpu_block = (gpu_block is None and stale_gpu_block is not None
                              and stale_gpu_block.related_block is cpu_block
                              and stale_gpu_block.ref_cnt == 0 and
                              self.free_block_queue.contains(stale_gpu_block))
        if reusable_gpu_block:
            gpu_block = stale_gpu_block
            if self.free_block_queue.contains(gpu_block):
                self.free_block_queue.remove(gpu_block)
        elif gpu_block is None:
            if self.get_num_free_blocks() <= 0:
                raise ValueError(
                    "Cannot upload CPU KV cache block without a free GPU block"
                )
            gpu_block = self.free_block_queue.popleft()
            assert gpu_block.ref_cnt == 0
            if self.enable_caching:
                self._maybe_evict_cached_block(gpu_block)
        elif stale_gpu_block is not None and stale_gpu_block.related_block is cpu_block:
            stale_gpu_block.block_in_cpu = False
            stale_gpu_block.related_block = None

        assert gpu_block is not None
        if not reusable_gpu_block:
            self.add_h2d_swap_map(cpu_block.block_id, gpu_block.block_id)
        else:
            self.no_copy_restore_count += 1

        if block_hash is not None:
            gpu_block.block_hash = block_hash
            self.cached_block_hash_to_block[block_hash][
                gpu_block.block_id] = gpu_block

        gpu_block.block_in_cpu = True
        gpu_block.related_block = cpu_block
        if gpu_block.ref_cnt == 0:
            self.free_block_queue.append(gpu_block)
        cpu_block.related_block = gpu_block
        self._touch_cpu_shadow(cpu_block)
        if proactive_upload:
            self._mark_proactive_gpu_restore(gpu_block)
        return gpu_block

    def cpu_block_has_valid_gpu_copy(self, cpu_block: KVCacheBlock) -> bool:
        block_hash = cpu_block.block_hash
        stale_gpu_block = cpu_block.related_block
        return (block_hash is not None and stale_gpu_block is not None
                and stale_gpu_block.related_block is cpu_block
                and stale_gpu_block.block_hash == block_hash
                and self.cached_block_hash_to_block.get(block_hash, {}).get(
                    stale_gpu_block.block_id) is stale_gpu_block)

    def _finish_preservation(
        self,
        preservation_id: int,
        *,
        effective_location: Optional[str] = None,
    ) -> None:
        preservation = self._pending_preservations.pop(preservation_id, None)
        if preservation is None:
            return
        if (preservation.cpu_block_id is not None
                and self._cpu_preservation_ids.get(
                    preservation.cpu_block_id) == preservation_id):
            self._cpu_preservation_ids.pop(preservation.cpu_block_id, None)
        if (preservation.gpu_block_id is not None
                and self._gpu_preservation_ids.get(
                    preservation.gpu_block_id) == preservation_id):
            self._gpu_preservation_ids.pop(preservation.gpu_block_id, None)
        if effective_location is None:
            self.ineffective_reuse_blocks += 1
            self.ineffective_preservations_by_hash[
                preservation.block_hash] += 1
            return
        self.effective_reuse_blocks += 1
        if effective_location == "cpu":
            self.effective_cpu_reuse_blocks += 1
        else:
            self.effective_gpu_reuse_blocks += 1

    def _resolve_if_unreachable(self, preservation_id: int) -> None:
        preservation = self._pending_preservations.get(preservation_id)
        if (preservation is not None and preservation.cpu_block_id is None
                and preservation.gpu_block_id is None):
            self._finish_preservation(preservation_id)

    def _detach_cpu_preservation(self, cpu_block_id: int) -> None:
        preservation_id = self._cpu_preservation_ids.pop(cpu_block_id, None)
        if preservation_id is None:
            return
        preservation = self._pending_preservations.get(preservation_id)
        if (preservation is not None
                and preservation.cpu_block_id == cpu_block_id):
            preservation.cpu_block_id = None
        self._resolve_if_unreachable(preservation_id)

    def _detach_gpu_preservation(self, gpu_block_id: int) -> None:
        preservation_id = self._gpu_preservation_ids.pop(gpu_block_id, None)
        if preservation_id is None:
            return
        preservation = self._pending_preservations.get(preservation_id)
        if (preservation is not None
                and preservation.gpu_block_id == gpu_block_id):
            preservation.gpu_block_id = None
        self._resolve_if_unreachable(preservation_id)

    def _start_preservation(self, cpu_block: KVCacheBlock) -> None:
        block_hash = cpu_block.block_hash
        if block_hash is None:
            return
        self._detach_cpu_preservation(cpu_block.block_id)
        preservation_id = self._next_preservation_id
        self._next_preservation_id += 1
        self._pending_preservations[preservation_id] = _PendingPreservation(
            block_hash=block_hash,
            cpu_block_id=cpu_block.block_id,
        )
        self._cpu_preservation_ids[cpu_block.block_id] = preservation_id
        self.preserved_d2h_blocks += 1

    def _mark_proactive_gpu_restore(self, gpu_block: KVCacheBlock) -> None:
        block_hash = gpu_block.block_hash
        cpu_block = gpu_block.related_block
        if block_hash is None or cpu_block is None:
            return
        preservation_id = self._cpu_preservation_ids.get(cpu_block.block_id)
        preservation = (self._pending_preservations.get(preservation_id)
                        if preservation_id is not None else None)
        if preservation is None or preservation.block_hash != block_hash:
            return
        if (preservation.gpu_block_id == gpu_block.block_id
                and self._gpu_preservation_ids.get(
                    gpu_block.block_id) == preservation_id):
            return
        self._detach_gpu_preservation(gpu_block.block_id)
        preservation.gpu_block_id = gpu_block.block_id
        self._gpu_preservation_ids[gpu_block.block_id] = preservation_id
        self.proactive_gpu_published_blocks += 1

    def record_restored_gpu_hits(
        self,
        gpu_blocks: Iterable[KVCacheBlock],
    ) -> int:
        hits = 0
        for gpu_block in gpu_blocks:
            preservation_id = self._gpu_preservation_ids.get(
                gpu_block.block_id)
            preservation = (self._pending_preservations.get(preservation_id)
                            if preservation_id is not None else None)
            if (preservation is None
                    or preservation.block_hash != gpu_block.block_hash):
                continue
            self._finish_preservation(preservation_id,
                                      effective_location="gpu")
            hits += 1
        return hits

    def record_preserved_cpu_hits(
        self,
        cpu_blocks: Iterable[KVCacheBlock],
    ) -> int:
        hits = 0
        for cpu_block in cpu_blocks:
            preservation_id = self._cpu_preservation_ids.get(
                cpu_block.block_id)
            preservation = (self._pending_preservations.get(preservation_id)
                            if preservation_id is not None else None)
            if (preservation is None
                    or preservation.block_hash != cpu_block.block_hash):
                continue
            self._finish_preservation(preservation_id,
                                      effective_location="cpu")
            hits += 1
        return hits

    def record_prefix_lookup_hashes(
        self,
        block_hashes: Iterable[BlockHashType],
    ) -> None:
        parent: Optional[BlockHashType] = None
        depth = 0
        for block_hash in block_hashes:
            self.prefix_parent_by_hash.setdefault(block_hash, parent)
            self.prefix_depth_by_hash.setdefault(block_hash, depth)
            self.prefix_lookup_hits_by_hash[block_hash] += 1
            self.real_prefix_lookup_hit_blocks += 1
            parent = block_hash
            depth += 1

    def get_prefix_reuse_score(self, block_hash: BlockHashType) -> int:
        return self.prefix_lookup_hits_by_hash.get(block_hash, 0)

    def get_prefix_parent(
        self,
        block_hash: BlockHashType,
    ) -> tuple[bool, Optional[BlockHashType]]:
        if block_hash not in self.prefix_parent_by_hash:
            return False, None
        return True, self.prefix_parent_by_hash[block_hash]

    def get_prefix_depth(self, block_hash: BlockHashType) -> int:
        return self.prefix_depth_by_hash.get(block_hash, 1 << 30)

    def get_ineffective_preservation_count(
        self,
        block_hash: BlockHashType,
    ) -> int:
        return self.ineffective_preservations_by_hash.get(block_hash, 0)

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        self._detach_gpu_preservation(block.block_id)
        return super()._maybe_evict_cached_block(block)

    def get_restore_feedback(self) -> dict[str, float | int]:
        resolved = self.effective_reuse_blocks + self.ineffective_reuse_blocks
        hit_rate = (self.effective_reuse_blocks /
                    resolved if resolved else 0.0)
        return {
            "preserved_d2h_blocks": self.preserved_d2h_blocks,
            "proactive_gpu_published_blocks":
            self.proactive_gpu_published_blocks,
            "pending_blocks": len(self._pending_preservations),
            "effective_reuse_blocks": self.effective_reuse_blocks,
            "effective_cpu_reuse_blocks": self.effective_cpu_reuse_blocks,
            "effective_gpu_reuse_blocks": self.effective_gpu_reuse_blocks,
            "ineffective_reuse_blocks": self.ineffective_reuse_blocks,
            "resolved_blocks": resolved,
            "effective_reuse_rate": hit_rate,
        }

    def protect_d2h_gpu_blocks(
        self,
        blocks: Iterable[KVCacheBlock],
    ) -> int:
        protected = 0
        for block in blocks:
            if block.block_id in self._d2h_protected_gpu_blocks:
                continue
            if block.ref_cnt != 0 or not self.free_block_queue.contains(block):
                raise ValueError(
                    f"Cannot protect non-free D2H source block {block.block_id}"
                )
            prev_id = (block.prev_free_block.block_id
                       if block.prev_free_block is not None else None)
            next_id = (block.next_free_block.block_id
                       if block.next_free_block is not None else None)
            self.free_block_queue.remove(block)
            self._d2h_protected_gpu_blocks[block.block_id] = (block, prev_id,
                                                              next_id)
            protected += 1
        return protected

    def release_d2h_gpu_blocks(self, block_ids: Iterable[int]) -> int:
        pending = {
            int(block_id):
            self._d2h_protected_gpu_blocks.pop(int(block_id), None)
            for block_id in block_ids
        }
        pending = {
            block_id: record
            for block_id, record in pending.items() if record is not None
        }
        for block, _, _ in pending.values():
            if block.ref_cnt != 0:
                raise RuntimeError(
                    f"Protected D2H source block {block.block_id} was allocated"
                )

        released = 0
        while pending:
            progressed = False
            for block_id, (block, prev_id, next_id) in tuple(pending.items()):
                next_block = (self.blocks[next_id]
                              if next_id is not None else None)
                prev_block = (self.blocks[prev_id]
                              if prev_id is not None else None)
                if (next_block is not None
                        and self.free_block_queue.contains(next_block)):
                    self.free_block_queue.insert_before(block, next_block)
                elif next_id is None:
                    self.free_block_queue.append(block)
                elif (prev_block is not None
                      and self.free_block_queue.contains(prev_block)):
                    self.free_block_queue.insert_after(block, prev_block)
                else:
                    continue
                pending.pop(block_id)
                released += 1
                progressed = True
            if progressed:
                continue
            for block_id, (block, _, _) in tuple(pending.items()):
                self.free_block_queue.append(block)
                pending.pop(block_id)
                released += 1
        return released

    def publish_completed_d2h_gpu_blocks(
        self,
        block_ids: Iterable[int],
    ) -> int:
        published = 0
        for block_id in block_ids:
            gpu_block = self.blocks[int(block_id)]
            cpu_block = gpu_block.related_block
            block_hash = gpu_block.block_hash
            if (block_hash is None or cpu_block is None
                    or cpu_block.related_block is not gpu_block
                    or cpu_block.block_hash != block_hash):
                continue
            self.cached_block_hash_to_block[block_hash][
                gpu_block.block_id] = gpu_block
            published += 1
        return published

    def get_imminent_eviction_block_ids(
        self,
        limit: int,
        excluded_block_ids: Optional[set[int]] = None,
    ) -> set[int]:
        block_ids: set[int] = set()
        excluded_block_ids = excluded_block_ids or set()
        block = self.free_block_queue.free_list_head
        while block is not None and len(block_ids) < max(0, int(limit)):
            if block.block_id not in excluded_block_ids:
                block_ids.add(block.block_id)
            block = block.next_free_block
        return block_ids

    def get_new_blocks(
        self,
        num_blocks: int,
        computed_cpu_blocks: Optional[list[KVCacheBlock]] = None,
        excluded_h2d_gpu_block_ids: Optional[set[int]] = None,
    ) -> list[KVCacheBlock]:
        if not computed_cpu_blocks:
            return super().get_new_blocks(num_blocks)

        computed_cpu_blocks = computed_cpu_blocks or []
        computed_cpu_blocks_len = len(computed_cpu_blocks)
        excluded_h2d_gpu_block_ids = excluded_h2d_gpu_block_ids or set()

        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the gpu pool")

        ret: list[KVCacheBlock] = []
        temporarily_protected = [
            self.blocks[block_id] for block_id in excluded_h2d_gpu_block_ids
            if self.blocks[block_id].ref_cnt == 0
            and self.free_block_queue.contains(self.blocks[block_id])
        ]
        self.protect_d2h_gpu_blocks(temporarily_protected)
        try:
            for cpu_block in computed_cpu_blocks:
                curr_block = self.upload_block(cpu_block)
                assert curr_block is not None
                if self.free_block_queue.contains(curr_block):
                    self.free_block_queue.remove(curr_block)
                curr_block.incr_ref()
                ret.append(curr_block)
        finally:
            self.release_d2h_gpu_blocks(block.block_id
                                        for block in temporarily_protected)

        idx = computed_cpu_blocks_len
        while idx < num_blocks:
            curr_block = self.free_block_queue.popleft()
            assert curr_block.ref_cnt == 0, f"curr_block {curr_block.block_id} ref_cnt: {curr_block.ref_cnt}"

            if self.enable_caching:
                self._maybe_evict_cached_block(curr_block)

            curr_block.block_in_cpu = False
            curr_block.related_block = None

            curr_block.incr_ref()

            ret.append(curr_block)
            idx += 1

        return ret

    def _maybe_evict_cached_cpu_block(self, cpu_block: KVCacheBlock) -> bool:
        block_hash = cpu_block.block_hash
        if block_hash and block_hash in self.cached_block_hash_to_cpu_block:
            self._detach_cpu_preservation(cpu_block.block_id)
            self.cpu_evict_count += 1
            self._drop_cpu_shadow(cpu_block)
            self._remove_cached_cpu_block_mapping(cpu_block, block_hash)
            if cpu_block.related_block is not None:
                cpu_block.related_block.block_in_cpu = False
                cpu_block.related_block.related_block = None
            cpu_block.reset_hash()
            return True
        return False

    def get_cached_cpu_block(
        self,
        block_hash: BlockHashType,
        excluded_cpu_block_ids: Optional[set[int]] = None,
    ) -> Optional[KVCacheBlock]:
        cached_cpu_blocks = self.cached_block_hash_to_cpu_block.get(block_hash)
        if not cached_cpu_blocks:
            return None
        excluded_cpu_block_ids = excluded_cpu_block_ids or set()
        for cpu_block in cached_cpu_blocks.values():
            if cpu_block.block_id not in excluded_cpu_block_ids:
                self._touch_cpu_shadow(cpu_block)
                return cpu_block
        return None

    def peek_cpu_shadow_block(
        self,
        block_hash: BlockHashType,
        excluded_cpu_block_ids: Optional[set[int]] = None,
    ) -> Optional[KVCacheBlock]:
        excluded_cpu_block_ids = excluded_cpu_block_ids or set()
        for cpu_block in self.cached_block_hash_to_cpu_block.get(
                block_hash, {}).values():
            if cpu_block.block_id not in excluded_cpu_block_ids:
                return cpu_block
        return None

    def touch(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            if block.related_block is not None:
                if block.related_block.ref_cnt == 0:
                    self.free_cpu_block_queue.remove(block.related_block)
                block.related_block.incr_ref()
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
            block.incr_ref()

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        for block in ordered_blocks:
            if block.related_block is not None:
                if block.related_block.ref_cnt == 0:
                    self.free_cpu_block_queue.append(block.related_block)
                elif block.related_block.ref_cnt > 0:
                    self.free_cpu_blocks([block.related_block])
            if block.ref_cnt == 0:
                self.free_block_queue.append(block)
                continue
            block.decr_ref()
            if block.ref_cnt == 0:
                self.free_block_queue.append(block)

    def free_cpu_blocks(self,
                        ordered_cpu_blocks: Iterable[KVCacheBlock]) -> None:
        for cpu_block in ordered_cpu_blocks:
            if cpu_block.ref_cnt > 0:
                cpu_block.decr_ref()
            if cpu_block.ref_cnt == 0:
                self.free_cpu_block_queue.append(cpu_block)
                self._register_cpu_shadow(cpu_block)

    def get_num_free_cpu_blocks(self) -> int:
        return self.free_cpu_block_queue.num_free_blocks

    def get_cpu_usage(self) -> float:
        return 1.0 - (self.get_num_free_cpu_blocks() / self.num_cpu_blocks)

    def get_cpu_evict_count(self) -> int:
        return self.cpu_evict_count

    def add_d2h_swap_map(self, gpu_block_id: int, cpu_block_id: int) -> None:
        self.step_d2h_swap_map[gpu_block_id] = cpu_block_id
        self.swap_out_count += 1
        self.d2h_logical_blocks += 1

    def add_h2d_swap_map(self, cpu_block_id: int, gpu_block_id: int) -> None:
        self.step_h2d_swap_map[cpu_block_id] = gpu_block_id
        self.swap_in_count += 1
        self.h2d_logical_blocks += 1

    def clear_step_d2h_swap_map(self) -> None:
        self.step_d2h_swap_map.clear()

    def clear_step_h2d_swap_map(self) -> None:
        self.step_h2d_swap_map.clear()
