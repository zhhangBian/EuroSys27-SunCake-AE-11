import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from vllm.logger import init_logger
from vllm.mcp.agent_info import AgentInfoManager
from vllm.utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.request import Request

logger = init_logger(__name__)

def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


MAX_RESERVE_RATIO = _env_float("VLLM_AGENT_MAX_RESERVE_RATIO", 0.3)
MIN_RESERVE_RATIO = _env_float("VLLM_AGENT_MIN_RESERVE_RATIO", 0.05)

IMPORTANT_AGENT_RATIO = _env_float("VLLM_AGENT_IMPORTANT_AGENT_RATIO", 0.75)

ADJUSTMENT_WINDOW = 500
GPU_USAGE_HIGH_WATERMARK = _env_float("VLLM_AGENT_GPU_USAGE_HIGH_WATERMARK",
                                      0.75)
GPU_USAGE_LOW_WATERMARK = _env_float("VLLM_AGENT_GPU_USAGE_LOW_WATERMARK",
                                     0.40)
RESERVE_ADJUSTMENT_STEP = _env_float("VLLM_AGENT_RESERVE_ADJUSTMENT_STEP",
                                     0.05)


@dataclass
class AdmissionDecision:
    allowed: bool
    demand_blocks: int
    shared_blocks_used: int = 0
    reserved_blocks_used: int = 0
    borrowed_reserved_blocks: int = 0
    reserved_usage_by_agent: dict[str, int] = field(default_factory=dict)
    reason: str = ""
    source: str = "none"


class AgentScheduler:

    def __init__(
        self,
        enable_agent_scheduling: bool,
        agent_info_manager: AgentInfoManager,
        num_gpu_blocks: int,
        block_size: int,
        kv_cache_manager: KVCacheManager,
    ):
        self.enable_agent_scheduling: bool = enable_agent_scheduling
        self.agent_info_manager: AgentInfoManager = agent_info_manager
        self.kv_cache_manager: KVCacheManager = kv_cache_manager

        self.num_gpu_blocks: int = num_gpu_blocks
        self.block_size: int = block_size

        self.agent_block_num_dict: dict[str, int] = {}
        self.agent_reserve_ratio_dict: dict[str, float] = {}
        self.agent_reserve_num_dict: dict[str, int] = {}
        self.agent_deferral_count: dict[str, int] = {}
        self.agent_preemption_count: dict[str, int] = {}

        self.adjustment_window = ADJUSTMENT_WINDOW
        self.schedule_counter = 0

        self.reserve_ratio = MIN_RESERVE_RATIO
        self.reserved_gpu_blocks = int(num_gpu_blocks * self.reserve_ratio)
        self.shared_gpu_blocks = num_gpu_blocks - self.reserved_gpu_blocks

        self.important_agent_types: set[str] = set()
        self.agent_scores: dict[str, float] = {}
        self._observed_priority_by_agent: dict[str, float] = {}

        self._waiting_count_by_agent: dict[str, int] = {}
        self._waiting_avg_wait_by_agent: dict[str, float] = {}
        self._shared_blocks_in_use = 0
        self._reserved_blocks_in_use_by_agent: dict[str, int] = {}
        self._request_shared_usage: dict[str, int] = {}
        self._request_reserved_usage: dict[str, dict[str, int]] = {}
        self._step_shared_available = self.shared_gpu_blocks
        self._step_reserved_available: dict[str, int] = {}
        self._step_waiting_critical_types: set[str] = set()
        self._reservation_plan_initialized = False

        logger.info(
            "[AgentScheduler] total gpu blocks: %s, reserve ratio: %.3f",
            num_gpu_blocks,
            self.reserve_ratio,
        )

    def begin_schedule_step(self, waiting_requests,
                            running_requests: list[Request]) -> None:
        if not self.enable_agent_scheduling:
            return

        waiting_list = list(waiting_requests)
        running_list = list(running_requests)

        self.schedule_counter += 1
        self._snapshot_waiting_state(waiting_list)

        if (not self._reservation_plan_initialized
                or self.schedule_counter % self.adjustment_window == 0):
            self._refresh_reservation_plan(waiting_list, running_list)

        (self._step_shared_available, self._step_reserved_available,
         _) = self._compute_step_capacity()
        self._step_waiting_critical_types = {
            request.agent_type
            for request in waiting_list
            if request.agent_type in self.important_agent_types
        }

    def estimate_incremental_gpu_blocks(
        self,
        req: Request,
        num_new_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
        new_computed_cpu_blocks: Optional[list[KVCacheBlock]] = None,
        num_lookahead_tokens: int = 0,
    ) -> int:
        new_computed_blocks = new_computed_blocks or []
        new_computed_cpu_blocks = new_computed_cpu_blocks or []

        req_blocks = self.kv_cache_manager.req_to_blocks[req.request_id]
        unique_new_computed_blocks = [
            blk for blk in new_computed_blocks if blk not in req_blocks
        ]

        num_computed_tokens = (
            req.num_computed_tokens +
            (len(unique_new_computed_blocks) + len(new_computed_cpu_blocks)) *
            self.block_size)
        total_tokens = num_computed_tokens + num_new_tokens + num_lookahead_tokens
        num_required_blocks = cdiv(total_tokens, self.block_size)
        num_new_blocks = (num_required_blocks - len(req_blocks) -
                          len(unique_new_computed_blocks))
        return max(0, num_new_blocks)

    def can_schedule_req(
        self,
        req: Request,
        num_new_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
        new_computed_cpu_blocks: Optional[list[KVCacheBlock]] = None,
        num_lookahead_tokens: int = 0,
    ) -> bool:
        if not self.enable_agent_scheduling:
            return True

        demand_blocks = self.estimate_incremental_gpu_blocks(
            req,
            num_new_tokens,
            new_computed_blocks=new_computed_blocks,
            new_computed_cpu_blocks=new_computed_cpu_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
        )
        decision = self._evaluate_admission(req, demand_blocks, mutate=False)
        return decision.allowed

    def record_request_new_blocks(self, req: Request,
                                  kvc_blocks: list[KVCacheBlock]):
        agent_type = req.agent_type
        allocated_blocks = len(kvc_blocks)
        if self.enable_agent_scheduling:
            decision = self._evaluate_admission(req,
                                                allocated_blocks,
                                                mutate=True)
            if decision.allowed:
                self._shared_blocks_in_use += decision.shared_blocks_used
                self._request_shared_usage[req.request_id] = (
                    self._request_shared_usage.get(req.request_id, 0) +
                    decision.shared_blocks_used)

                if decision.reserved_usage_by_agent:
                    request_reserved_usage = self._request_reserved_usage.setdefault(
                        req.request_id, {})
                    for agent_type, used_blocks in (
                            decision.reserved_usage_by_agent.items()):
                        self._reserved_blocks_in_use_by_agent[agent_type] = (
                            self._reserved_blocks_in_use_by_agent.get(
                                agent_type, 0) + used_blocks)
                        request_reserved_usage[agent_type] = (
                            request_reserved_usage.get(agent_type, 0) +
                            used_blocks)
            else:
                logger.warning(
                    "[AgentScheduler] allocated KV blocks for %s but admission accounting rejected the request: %s",
                    req.request_id,
                    decision.reason,
                )
        if agent_type:
            self.agent_block_num_dict[agent_type] = (
                self.agent_block_num_dict.get(agent_type, 0) + allocated_blocks)
            self.agent_deferral_count[agent_type] = max(
                0,
                self.agent_deferral_count.get(agent_type, 0) - 1,
            )

    def finish_request(self, req: Request, kvc_blocks: list[KVCacheBlock]):
        agent_type = req.agent_type
        shared_usage = self._request_shared_usage.pop(req.request_id, 0)
        self._shared_blocks_in_use = max(0,
                                         self._shared_blocks_in_use -
                                         shared_usage)

        reserved_usage = self._request_reserved_usage.pop(req.request_id, {})
        for reserve_owner, used_blocks in reserved_usage.items():
            remaining = max(
                0,
                self._reserved_blocks_in_use_by_agent.get(reserve_owner, 0) -
                used_blocks,
            )
            if remaining == 0:
                self._reserved_blocks_in_use_by_agent.pop(reserve_owner, None)
            else:
                self._reserved_blocks_in_use_by_agent[reserve_owner] = remaining

        if agent_type:
            self.agent_block_num_dict[agent_type] = max(
                0,
                self.agent_block_num_dict.get(agent_type, 0) - len(kvc_blocks),
            )

    def record_request_deferred(self, req: Request) -> None:
        agent_type = req.agent_type
        if not agent_type:
            return
        self.agent_deferral_count[agent_type] = (
            self.agent_deferral_count.get(agent_type, 0) + 1)

    def record_request_preempted(self, req: Request) -> None:
        agent_type = req.agent_type
        if not agent_type:
            return
        self.agent_preemption_count[agent_type] = (
            self.agent_preemption_count.get(agent_type, 0) + 1)

    def _snapshot_waiting_state(self, waiting_requests: list[Request]) -> None:
        now = time.time()
        wait_sum_by_agent: dict[str, float] = {}
        wait_count_by_agent: dict[str, int] = {}
        for request in waiting_requests:
            agent_type = request.agent_type
            if not agent_type:
                continue
            wait_sum_by_agent[agent_type] = wait_sum_by_agent.get(
                agent_type, 0.0) + max(0.0, now - request.arrival_time)
            wait_count_by_agent[agent_type] = wait_count_by_agent.get(
                agent_type, 0) + 1

        self._waiting_count_by_agent = wait_count_by_agent
        self._waiting_avg_wait_by_agent = {
            agent_type: wait_sum_by_agent[agent_type] / count
            for agent_type, count in wait_count_by_agent.items()
            if count > 0
        }

    def _refresh_reservation_plan(self, waiting_requests: list[Request],
                                  running_requests: list[Request]) -> None:
        self._observed_priority_by_agent = dict(
            self.agent_info_manager.get_agent_priority_dict())
        for request in waiting_requests + running_requests:
            if not request.agent_type:
                continue
            self._observed_priority_by_agent[request.agent_type] = max(
                request.priority,
                self._observed_priority_by_agent.get(request.agent_type, 0.0),
            )

        all_agent_types = self._collect_agent_types(waiting_requests,
                                                    running_requests)

        self.agent_scores = {}
        for agent_type in all_agent_types:
            self.agent_scores[agent_type] = self._compute_agent_score(agent_type)

        self.important_agent_types = self._select_critical_agent_types(
            all_agent_types)
        self._update_reserve_ratio_from_gpu_usage()
        self._partition_reserved_blocks()

        self._reservation_plan_initialized = True

    def _collect_agent_types(self, waiting_requests: list[Request],
                             running_requests: list[Request]) -> set[str]:
        agent_types = set(self.agent_info_manager.get_agent_priority_dict())
        agent_types.update(
            agent_type for agent_type, count in self.agent_block_num_dict.items()
            if count > 0)
        agent_types.update(
            request.agent_type for request in waiting_requests
            if request.agent_type)
        agent_types.update(
            request.agent_type for request in running_requests
            if request.agent_type)
        agent_types.update(self._waiting_count_by_agent)
        return agent_types

    def _compute_agent_score(self, agent_type: str) -> float:
        static_priority = float(
            self._observed_priority_by_agent.get(
                agent_type,
                self.agent_info_manager.get_agent_priority_dict().get(
                    agent_type, 0.0),
            ))

        avg_wait = self._waiting_avg_wait_by_agent.get(agent_type, 0.0)
        waiting_count = self._waiting_count_by_agent.get(agent_type, 0)
        deferrals = self.agent_deferral_count.get(agent_type, 0)
        preemptions = self.agent_preemption_count.get(agent_type, 0)
        runtime_urgency = (math.log1p(avg_wait) + math.log1p(waiting_count) +
                           math.log1p(deferrals) +
                           1.5 * math.log1p(preemptions))

        avg_input_len, avg_output_len, avg_time = (
            self.agent_info_manager.get_ave_past_info(agent_type))
        historical_tokens = max(0.0, avg_input_len + avg_output_len)
        if avg_time > 0 and historical_tokens > 0:
            throughput = historical_tokens / avg_time
            historical_cost = (math.log1p(historical_tokens) +
                               math.log1p(avg_time) +
                               math.log1p(max(0.0, throughput)))
        elif historical_tokens > 0:
            historical_cost = math.log1p(historical_tokens)
        else:
            historical_cost = 0.0

        graph_hint = 0.0
        type_infos = self.agent_info_manager._get_agent_type_list(agent_type)
        if type_infos:
            avg_depth = sum(info.depth for info in type_infos) / len(type_infos)
            avg_out_degree = sum(info.out_degree for info in type_infos) / len(type_infos)
            avg_in_degree = sum(info.in_degree for info in type_infos) / len(type_infos)
            avg_app_elapsed = sum(info.app_elapsed_time for info in type_infos) / len(type_infos)
            avg_remaining_depth = sum(info.remaining_depth for info in type_infos) / len(type_infos)
            graph_hint = (0.7 * avg_depth + 0.5 * avg_out_degree +
                          0.25 * avg_in_degree +
                          0.20 * avg_remaining_depth +
                          0.10 * min(avg_app_elapsed, 180.0))
        return max(1.0, (2.0 * static_priority) + runtime_urgency +
                   (0.5 * historical_cost) + graph_hint)

    def _select_critical_agent_types(self,
                                     all_agent_types: set[str]) -> set[str]:
        if not all_agent_types:
            return set()

        sorted_agents = sorted(
            all_agent_types,
            key=lambda agent_type: (
                self.agent_scores.get(agent_type, 1.0),
                self._observed_priority_by_agent.get(
                    agent_type,
                    self.agent_info_manager.get_agent_priority_dict().get(
                        agent_type, 0.0),
                ),
                agent_type,
            ),
            reverse=True,
        )
        top_n = max(1, int(len(sorted_agents) * IMPORTANT_AGENT_RATIO))
        return set(sorted_agents[:top_n])

    def _update_reserve_ratio_from_gpu_usage(self) -> None:
        usage = self.kv_cache_manager.usage
        if usage >= GPU_USAGE_HIGH_WATERMARK:
            self.reserve_ratio += RESERVE_ADJUSTMENT_STEP
        elif usage <= GPU_USAGE_LOW_WATERMARK:
            self.reserve_ratio -= RESERVE_ADJUSTMENT_STEP

        self.reserve_ratio = min(MAX_RESERVE_RATIO,
                                 max(MIN_RESERVE_RATIO, self.reserve_ratio))
        self.reserved_gpu_blocks = int(self.num_gpu_blocks * self.reserve_ratio)
        self.shared_gpu_blocks = max(0,
                                     self.num_gpu_blocks -
                                     self.reserved_gpu_blocks)

    def _partition_reserved_blocks(self) -> None:
        self.agent_reserve_ratio_dict = {}
        self.agent_reserve_num_dict = {}

        if not self.important_agent_types or self.reserved_gpu_blocks <= 0:
            return

        total_active_blocks = sum(
            max(0, count) for count in self.agent_block_num_dict.values())
        total_score = sum(
            self.agent_scores.get(agent_type, 1.0)
            for agent_type in self.important_agent_types)

        raw_weights: dict[str, float] = {}
        for agent_type in self.important_agent_types:
            mem_ratio = (
                self.agent_block_num_dict.get(agent_type, 0) / total_active_blocks
                if total_active_blocks > 0 else 0.0)
            score_ratio = (
                self.agent_scores.get(agent_type, 1.0) / total_score
                if total_score > 0 else 0.0)
            raw_weights[agent_type] = max(score_ratio,
                                          (mem_ratio + score_ratio) / 2.0,
                                          1e-6)

        total_weight = sum(raw_weights.values())
        normalized_weights = {
            agent_type: weight / total_weight
            for agent_type, weight in raw_weights.items()
        }

        allocated = 0
        remainders: list[tuple[float, str]] = []
        for agent_type, weight in normalized_weights.items():
            raw_block_count = weight * self.reserved_gpu_blocks
            block_count = int(math.floor(raw_block_count))
            self.agent_reserve_num_dict[agent_type] = block_count
            allocated += block_count
            remainders.append((raw_block_count - block_count, agent_type))

        remainder = self.reserved_gpu_blocks - allocated
        for _, agent_type in sorted(remainders, reverse=True)[:remainder]:
            self.agent_reserve_num_dict[agent_type] += 1

        self.agent_reserve_ratio_dict = {
            agent_type: block_count / self.num_gpu_blocks
            for agent_type, block_count in self.agent_reserve_num_dict.items()
        }

    def _compute_step_capacity(self) -> tuple[int, dict[str, int], int]:
        reserved_available = {
            agent_type:
            max(
                0,
                reserved_blocks -
                self._reserved_blocks_in_use_by_agent.get(agent_type, 0),
            )
            for agent_type, reserved_blocks in self.agent_reserve_num_dict.items()
        }
        shared_available = max(0,
                               self.shared_gpu_blocks -
                               self._shared_blocks_in_use)
        physical_free_blocks = max(
            0,
            self.num_gpu_blocks -
            (self._shared_blocks_in_use +
             sum(self._reserved_blocks_in_use_by_agent.values())),
        )

        total_available = shared_available + sum(reserved_available.values())
        excess = max(0, total_available - physical_free_blocks)
        if excess > 0:
            shared_reduction = min(shared_available, excess)
            shared_available -= shared_reduction
            excess -= shared_reduction
        if excess > 0:
            self._reduce_reserved_capacity(reserved_available, excess)

        return shared_available, reserved_available, physical_free_blocks

    def _evaluate_admission(self,
                            request: Request,
                            demand_blocks: int,
                            *,
                            mutate: bool) -> AdmissionDecision:
        if demand_blocks <= 0:
            return AdmissionDecision(
                allowed=True,
                demand_blocks=0,
                reason="no-new-blocks",
                source="none",
            )

        shared_blocks_used = min(self._step_shared_available, demand_blocks)
        remaining = demand_blocks - shared_blocks_used
        agent_type = request.agent_type
        is_critical = agent_type in self.important_agent_types
        reserved_usage_by_agent: dict[str, int] = {}

        if remaining <= 0:
            if mutate:
                self._step_shared_available -= shared_blocks_used
            return AdmissionDecision(
                allowed=True,
                demand_blocks=demand_blocks,
                shared_blocks_used=shared_blocks_used,
                reserved_usage_by_agent=reserved_usage_by_agent,
                reason="fits-shared-pool",
                source="shared",
            )

        if is_critical:
            own_reserved = self._step_reserved_available.get(agent_type, 0)
            own_reserved_used = min(own_reserved, remaining)
            remaining -= own_reserved_used
            if own_reserved_used > 0:
                reserved_usage_by_agent[agent_type] = own_reserved_used

            borrowed_reserved_blocks = 0
            borrowed_usage_by_agent: dict[str, int] = {}
            if remaining > 0:
                borrowable_other_reserved = self._sum_reserved_available(
                    exclude={agent_type})
                if (not self._has_other_waiting_critical(agent_type)
                        and remaining <= borrowable_other_reserved):
                    borrowed_usage_by_agent = self._plan_reserved_consumption(
                        remaining, exclude={agent_type})
                    borrowed_reserved_blocks = sum(
                        borrowed_usage_by_agent.values())
                    remaining -= borrowed_reserved_blocks
                    for reserve_owner, used_blocks in (
                            borrowed_usage_by_agent.items()):
                        reserved_usage_by_agent[reserve_owner] = (
                            reserved_usage_by_agent.get(reserve_owner, 0) +
                            used_blocks)

            if remaining > 0:
                return AdmissionDecision(
                    allowed=False,
                    demand_blocks=demand_blocks,
                    shared_blocks_used=shared_blocks_used,
                    reserved_blocks_used=own_reserved_used,
                    borrowed_reserved_blocks=borrowed_reserved_blocks,
                    reserved_usage_by_agent=reserved_usage_by_agent,
                    reason="insufficient-protected-capacity",
                    source="denied",
                )

            if mutate:
                self._step_shared_available -= shared_blocks_used
                self._step_reserved_available[agent_type] = max(
                    0, own_reserved - own_reserved_used)
                if borrowed_reserved_blocks > 0:
                    self._consume_reserved_blocks(borrowed_usage_by_agent)

            return AdmissionDecision(
                allowed=True,
                demand_blocks=demand_blocks,
                shared_blocks_used=shared_blocks_used,
                reserved_blocks_used=own_reserved_used,
                borrowed_reserved_blocks=borrowed_reserved_blocks,
                reserved_usage_by_agent=reserved_usage_by_agent,
                reason="critical-request-protected",
                source=("reserved+borrowed"
                        if borrowed_reserved_blocks > 0 else "reserved"),
            )

        borrowable_reserved = self._sum_reserved_available()
        if (self._step_waiting_critical_types or remaining > borrowable_reserved):
            return AdmissionDecision(
                allowed=False,
                demand_blocks=demand_blocks,
                shared_blocks_used=shared_blocks_used,
                reserved_usage_by_agent=reserved_usage_by_agent,
                reason="reserved-capacity-protected",
                source="denied",
            )

        borrowed_usage_by_agent = self._plan_reserved_consumption(
            remaining, exclude=None)
        borrowed_reserved_blocks = sum(borrowed_usage_by_agent.values())
        reserved_usage_by_agent.update(borrowed_usage_by_agent)
        if mutate:
            self._step_shared_available -= shared_blocks_used
            self._consume_reserved_blocks(borrowed_usage_by_agent)

        return AdmissionDecision(
            allowed=True,
            demand_blocks=demand_blocks,
            shared_blocks_used=shared_blocks_used,
            borrowed_reserved_blocks=borrowed_reserved_blocks,
            reserved_usage_by_agent=reserved_usage_by_agent,
            reason="borrowed-idle-reserved-capacity",
            source="borrowed-reserved",
        )

    def _sum_reserved_available(self,
                                exclude: Optional[set[str]] = None) -> int:
        exclude = exclude or set()
        return sum(
            available for agent_type, available in
            self._step_reserved_available.items() if agent_type not in exclude)

    def _plan_reserved_consumption(
        self,
        block_count: int,
        exclude: Optional[set[str]],
    ) -> dict[str, int]:
        exclude = exclude or set()
        remaining = block_count
        usage_by_agent: dict[str, int] = {}
        donors = sorted(
            ((agent_type, available)
             for agent_type, available in self._step_reserved_available.items()
             if available > 0 and agent_type not in exclude),
            key=lambda item: (item[1], self.agent_scores.get(item[0], 1.0)),
            reverse=True,
        )

        for agent_type, available in donors:
            if remaining <= 0:
                break
            used = min(available, remaining)
            usage_by_agent[agent_type] = used
            remaining -= used
        return usage_by_agent

    def _consume_reserved_blocks(self,
                                 usage_by_agent: dict[str, int]) -> None:
        for agent_type, used_blocks in usage_by_agent.items():
            self._step_reserved_available[agent_type] = max(
                0,
                self._step_reserved_available.get(agent_type, 0) - used_blocks,
            )

    def _reduce_reserved_capacity(self, reserved_available: dict[str, int],
                                  excess_blocks: int) -> None:
        remaining = excess_blocks
        donors = sorted(
            reserved_available,
            key=lambda agent_type: self.agent_scores.get(agent_type, 1.0),
        )
        for agent_type in donors:
            if remaining <= 0:
                break
            reducible = min(reserved_available[agent_type], remaining)
            reserved_available[agent_type] -= reducible
            remaining -= reducible

    def _has_other_waiting_critical(self, agent_type: str) -> bool:
        return any(
            waiting_agent_type != agent_type
            for waiting_agent_type in self._step_waiting_critical_types)
