# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.cpu_offloading_block_pool import CpuOffloadingBlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock, hash_request_tokens
from vllm.v1.core.specialized_manager import get_specialized_manager
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)


class CpuOffloadingKVCacheManager(KVCacheManager):

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        num_preallocate_tokens: int = 64,
        log_stats: bool = False,
    ) -> None:
        logger.info("use CpuOffloadingKVCacheManager.")
        super().__init__(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            enable_caching=enable_caching,
            caching_hash_algo=caching_hash_algo,
            num_preallocate_tokens=num_preallocate_tokens,
            log_stats=log_stats,
        )
        self.num_cpu_blocks = kv_cache_config.num_cpu_blocks
        self.enable_caching = enable_caching
        self.block_pool = CpuOffloadingBlockPool(
            self.num_gpu_blocks,
            self.num_cpu_blocks,
            self.enable_caching,
        )
        kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.specialized_manager = get_specialized_manager(
            kv_cache_spec, self.block_pool)
        self.cpu_prefix_cache_stats: Optional[PrefixCacheStats] = (
            PrefixCacheStats() if log_stats else None
        )
        self._prefix_reuse_recorded_request_ids: set[str] = set()

    def get_computed_blocks(
            self, request: Request
    ) -> tuple[list[KVCacheBlock], list[KVCacheBlock], int]:
        return self._lookup_computed_blocks(request, record_stats=True)

    def peek_computed_blocks(
            self, request: Request
    ) -> tuple[list[KVCacheBlock], list[KVCacheBlock], int]:
        return self._lookup_computed_blocks(request, record_stats=False)

    def _lookup_computed_blocks(
        self,
        request: Request,
        *,
        record_stats: bool,
    ) -> tuple[list[KVCacheBlock], list[KVCacheBlock], int]:
        if not self.enable_caching:
            return [], [], 0

        block_hashes = self.req_to_block_hashes[request.request_id]
        if not block_hashes:
            block_hashes = hash_request_tokens(self.caching_hash_fn,
                                               self.block_size, request)
            self.req_to_block_hashes[request.request_id] = block_hashes

        if record_stats and self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1
        if request.sampling_params.prompt_logprobs is not None:
            return [], [], 0

        lookup_hashes = block_hashes
        if len(block_hashes) * self.block_size == request.num_tokens:
            lookup_hashes = block_hashes[:-1]

        computed_blocks = self.specialized_manager.find_longest_cache_hit(
            lookup_hashes)
        computed_cpu_blocks: list[KVCacheBlock] = []
        num_prefix_hits = len(computed_blocks)
        pending_d2h_cpu_block_ids = set(
            self.block_pool.step_d2h_swap_map.values())
        for block_hash in lookup_hashes[num_prefix_hits:]:
            if cached_gpu_block := self.block_pool.get_cached_block(block_hash):
                computed_blocks.append(cached_gpu_block)
            elif cached_cpu_block := self.block_pool.get_cached_cpu_block(
                    block_hash, pending_d2h_cpu_block_ids):
                computed_cpu_blocks.append(cached_cpu_block)
            else:
                break
            num_prefix_hits += 1

        if record_stats and self.log_stats:
            assert self.prefix_cache_stats is not None
            assert self.cpu_prefix_cache_stats is not None
            self.prefix_cache_stats.queries += len(block_hashes)
            self.prefix_cache_stats.hits += len(computed_blocks)
            self.cpu_prefix_cache_stats.queries += (
                len(block_hashes) - len(computed_blocks)
            )
            self.cpu_prefix_cache_stats.hits += len(computed_cpu_blocks)

        if record_stats:
            self.block_pool.record_restored_gpu_hits(computed_blocks)
            self.block_pool.record_preserved_cpu_hits(computed_cpu_blocks)
            if request.request_id not in self._prefix_reuse_recorded_request_ids:
                self.block_pool.record_prefix_lookup_hashes(
                    lookup_hashes[:num_prefix_hits])
                self._prefix_reuse_recorded_request_ids.add(request.request_id)

        num_computed_tokens = (len(computed_blocks) +
                               len(computed_cpu_blocks)) * self.block_size
        return computed_blocks, computed_cpu_blocks, num_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
        num_lookahead_tokens: int = 0,
        new_computed_cpu_blocks: Optional[list[KVCacheBlock]] = None,
        excluded_h2d_gpu_block_ids: Optional[set[int]] = None,
    ) -> Optional[list[KVCacheBlock]]:
        if num_tokens == 0:
            raise ValueError("num_tokens must be greater than 0")

        new_computed_blocks = new_computed_blocks or []
        new_computed_cpu_blocks = new_computed_cpu_blocks or []
        excluded_h2d_gpu_block_ids = excluded_h2d_gpu_block_ids or set()

        req_blocks = self.req_to_blocks[request.request_id]

        removed_blocks = self.specialized_manager.remove_skipped_blocks(
            req_blocks, request.num_computed_tokens)
        self.block_pool.free_blocks(removed_blocks)

        unique_new_computed_blocks = [
            blk for blk in new_computed_blocks if blk not in req_blocks
        ]

        num_computed_tokens = (
            request.num_computed_tokens +
            (len(unique_new_computed_blocks) + len(new_computed_cpu_blocks))
            * self.block_size
        )
        total_tokens = num_computed_tokens + num_tokens + num_lookahead_tokens
        num_required_blocks = cdiv(total_tokens, self.block_size)

        num_new_blocks = (num_required_blocks - len(req_blocks) - len(unique_new_computed_blocks))

        num_evictable_computed_blocks = sum(1 for blk in unique_new_computed_blocks
                                            if blk.ref_cnt == 0)
        if (num_new_blocks > self.block_pool.get_num_free_blocks() -
                num_evictable_computed_blocks):
            logger.debug(
                "allocate gpu blocks deferred for %s, need=%s free=%s evictable_computed=%s",
                request.request_id,
                num_new_blocks,
                self.block_pool.get_num_free_blocks(),
                num_evictable_computed_blocks,
            )
            return None
        excluded_free_gpu_blocks = sum(
            self.block_pool.free_block_queue.contains(
                self.block_pool.blocks[block_id])
            for block_id in excluded_h2d_gpu_block_ids)
        safe_h2d_targets = (
            self.block_pool.get_num_free_blocks() -
            num_evictable_computed_blocks - excluded_free_gpu_blocks)
        if len(new_computed_cpu_blocks) > safe_h2d_targets:
            logger.debug(
                "allocate H2D targets deferred for %s, need=%s safe_free=%s",
                request.request_id,
                len(new_computed_cpu_blocks),
                safe_h2d_targets,
            )
            return None

        if self.enable_caching:
            self.block_pool.touch(unique_new_computed_blocks)
            self.block_pool.touch_cpu_blocks(new_computed_cpu_blocks)
        else:
            assert not unique_new_computed_blocks, (
                "Computed blocks should be empty when "
                "prefix caching is disabled")
            assert not new_computed_cpu_blocks, (
                "Computed CPU blocks should be empty when "
                "prefix caching is disabled")

        req_blocks.extend(unique_new_computed_blocks)

        if num_new_blocks <= 0:
            new_blocks = []
        else:
            num_preallocate_blocks = max(
                0, self.num_preallocate_blocks -
                num_lookahead_tokens // self.block_size)
            num_new_blocks = min(
                num_new_blocks + num_preallocate_blocks,
                self.block_pool.get_num_free_blocks(),
                self.max_num_blocks_per_req - len(req_blocks),
            )
            assert num_new_blocks > 0

            new_blocks = self.block_pool.get_new_blocks(
                num_new_blocks,
                new_computed_cpu_blocks,
                excluded_h2d_gpu_block_ids,
            )
            req_blocks.extend(new_blocks)

        known_hashes = len(self.req_to_block_hashes[request.request_id])
        initial_cached_blocks = (len(unique_new_computed_blocks) +
                                 len(new_computed_cpu_blocks))
        num_cached_blocks = min(
            self.num_cached_block.get(request.request_id,
                                      initial_cached_blocks),
            known_hashes,
        )
        num_full_blocks_after_append = (
            num_computed_tokens + num_tokens - len(request.spec_token_ids)
        ) // self.block_size

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=req_blocks,
            block_hashes=self.req_to_block_hashes[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks_after_append,
            block_size=self.block_size,
            hash_fn=self.caching_hash_fn,
        )
        self.num_cached_block[request.request_id] = num_full_blocks_after_append
        return new_blocks

    def get_d2h_swap_map(self) -> dict[int, int]:
        return self.block_pool.step_d2h_swap_map.copy()

    def get_h2d_swap_map(self) -> dict[int, int]:
        return self.block_pool.step_h2d_swap_map.copy()

    def clear_step_d2h_swap_map(self) -> None:
        self.block_pool.clear_step_d2h_swap_map()

    def clear_step_h2d_swap_map(self) -> None:
        self.block_pool.clear_step_h2d_swap_map()

    def make_cpu_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
        if not self.log_stats:
            return None
        stats = self.cpu_prefix_cache_stats
        assert stats is not None
        self.cpu_prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_swap_in_count(self) -> int:
        return self.block_pool.swap_in_count

    def get_swap_out_count(self) -> int:
        return self.block_pool.swap_out_count

    @property
    def cpu_usage(self) -> float:
        return self.block_pool.get_cpu_usage()

    def get_gpu_cpu_evict_count(self) -> tuple[int, int]:
        return (
            self.block_pool.get_evict_count(),
            self.block_pool.get_cpu_evict_count()
        )
