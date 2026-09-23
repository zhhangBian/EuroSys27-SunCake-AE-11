# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional


@dataclass
class PressureSnapshot:
    step_id: int
    gpu_total_blocks: int
    gpu_free_blocks: int
    gpu_used_blocks: int
    gpu_usage: float
    gpu_clean_free_blocks: int = 0
    agent_shared_available: int = 0
    agent_reserved_available: dict[str, int] = field(default_factory=dict)
    waiting_demand_blocks: int = 0
    critical_waiting_demand_blocks: int = 0
    waiting_request_count: int = 0
    critical_waiting_request_count: int = 0
    offloadable_stalled_blocks: int = 0
    upload_debt_blocks: int = 0
    cpu_free_blocks: int = 0
    recent_swap_events: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class RequestImportance:
    request_id: str
    agent_type: str = ""
    static_priority: float = 0.0
    queue_priority: float = 0.0
    agent_type_score: float = 0.0
    normalized_score: float = 0.0
    is_critical: bool = False
    completion_ratio: float = 0.0
    remaining_depth: float = 0.0
    near_completion: bool = False


@dataclass
class OffloadDecision:
    request_id: str
    allowed: bool
    reason: str
    final_score: float
    predicted_duration: float
    transfer_time: float
    stall_margin: float
    pressure_score: float
    fit_score: float
    importance_penalty: float
    completion_penalty: float
    upload_safety_score: float
    cpu_capacity_score: float
    churn_penalty: float
    freed_blocks: int
    cpu_free_blocks: int
    queue_fit_request_id: Optional[str]
    queue_fit_demand_blocks: int
    request_importance: RequestImportance
    pressure: PressureSnapshot


class AgentOffloadCoordinator:

    def __init__(
        self,
        *,
        min_gpu_usage_for_offload: float = 0.60,
        high_pressure_gpu_usage: float = 0.85,
        offload_score_threshold: float = 1.0,
        high_importance_threshold: float = 8.0,
        near_completion_remaining_depth: float = 1.0,
        churn_cooldown_steps: int = 4,
        global_churn_budget: int = 64,
    ) -> None:
        self.min_gpu_usage_for_offload = min_gpu_usage_for_offload
        self.high_pressure_gpu_usage = high_pressure_gpu_usage
        self.offload_score_threshold = offload_score_threshold
        self.high_importance_threshold = high_importance_threshold
        self.near_completion_remaining_depth = near_completion_remaining_depth
        self.churn_cooldown_steps = churn_cooldown_steps
        self.global_churn_budget = global_churn_budget

        self.step_id = 0
        self.recent_offload_step_by_request: dict[str, int] = {}
        self._recent_swap_steps: list[int] = []

    def begin_step(self) -> int:
        self.step_id += 1
        self._recent_swap_steps = [
            step for step in self._recent_swap_steps
            if self.step_id - step <= self.churn_cooldown_steps
        ]
        return self.step_id

    def build_pressure_snapshot(
        self,
        *,
        gpu_total_blocks: int,
        gpu_free_blocks: int,
        gpu_usage: float,
        waiting_requests: Iterable[Any],
        block_size: int,
        cpu_free_blocks: int = 0,
        agent_scheduler: Any = None,
        mcp_manager: Any = None,
        upload_debt_blocks: int = 0,
        offloadable_stalled_blocks: int = 0,
        gpu_clean_free_blocks: Optional[int] = None,
    ) -> PressureSnapshot:
        waiting_list = list(waiting_requests)

        important_agent_types = set(
            getattr(agent_scheduler, "important_agent_types", set()) or set())
        agent_reserved_available = dict(
            getattr(agent_scheduler, "_step_reserved_available", {}) or {})
        agent_shared_available = int(
            getattr(agent_scheduler, "_step_shared_available", 0) or 0)

        waiting_demand_blocks = 0
        critical_waiting_demand_blocks = 0
        critical_waiting_count = 0
        for request in waiting_list:
            remaining_tokens = max(
                0,
                int(getattr(request, "num_tokens", 0) or 0) -
                int(getattr(request, "num_computed_tokens", 0) or 0),
            )
            demand = int(math.ceil(remaining_tokens / block_size)) if block_size else 0
            waiting_demand_blocks += demand
            if getattr(request, "agent_type", "") in important_agent_types:
                critical_waiting_count += 1
                critical_waiting_demand_blocks += demand

        if mcp_manager is not None and offloadable_stalled_blocks <= 0:
            for req_id in getattr(mcp_manager, "requests_to_offload", set()):
                offloadable_stalled_blocks += len(
                    mcp_manager.get_finished_request_blocks(req_id))

        gpu_total_blocks = max(0, int(gpu_total_blocks))
        gpu_free_blocks = max(0, int(gpu_free_blocks))
        gpu_used_blocks = max(0, gpu_total_blocks - gpu_free_blocks)
        snapshot = PressureSnapshot(
            step_id=self.step_id,
            gpu_total_blocks=gpu_total_blocks,
            gpu_free_blocks=gpu_free_blocks,
            gpu_used_blocks=gpu_used_blocks,
            gpu_usage=max(0.0, min(1.0, float(gpu_usage or 0.0))),
            gpu_clean_free_blocks=max(
                0,
                int(gpu_free_blocks if gpu_clean_free_blocks is None else
                    gpu_clean_free_blocks),
            ),
            agent_shared_available=agent_shared_available,
            agent_reserved_available=agent_reserved_available,
            waiting_demand_blocks=waiting_demand_blocks,
            critical_waiting_demand_blocks=critical_waiting_demand_blocks,
            waiting_request_count=len(waiting_list),
            critical_waiting_request_count=critical_waiting_count,
            offloadable_stalled_blocks=offloadable_stalled_blocks,
            upload_debt_blocks=max(0, int(upload_debt_blocks)),
            cpu_free_blocks=max(0, int(cpu_free_blocks)),
            recent_swap_events=len(self._recent_swap_steps),
        )
        return snapshot

    def get_request_importance(
        self,
        request_id: str,
        *,
        request: Any = None,
        metadata: Optional[dict[str, Any]] = None,
        agent_scheduler: Any = None,
        priority_getter: Optional[Callable[[Any], float]] = None,
    ) -> RequestImportance:
        metadata = dict(metadata or getattr(request, "agent_info", {}) or {})
        agent_type = str(metadata.get("type", getattr(request, "agent_type", "")) or "")
        static_priority = self._safe_float(
            metadata.get("priority", getattr(request, "priority", 0.0)))

        queue_priority = 0.0
        if request is not None and priority_getter is not None:
            try:
                queue_priority = self._safe_float(priority_getter(request))
            except Exception:
                queue_priority = 0.0

        agent_type_score = 0.0
        if agent_scheduler is not None and agent_type:
            agent_type_score = self._safe_float(
                getattr(agent_scheduler, "agent_scores", {}).get(agent_type, 0.0))

        queue_component = math.log1p(max(0.0, queue_priority))
        agent_component = min(10.0, agent_type_score / 2.0)
        normalized_score = max(0.0, min(10.0, max(
            static_priority,
            queue_component,
            agent_component,
        )))

        important_agent_types = set(
            getattr(agent_scheduler, "important_agent_types", set()) or set())
        is_critical = bool(
            (agent_type and agent_type in important_agent_types)
            or metadata.get("critical_path", False))

        max_depth = max(0.0, self._safe_float(metadata.get("app_max_depth", 0.0)))
        depth = max(0.0, self._safe_float(metadata.get("depth", 0.0)))
        remaining_depth = self._safe_float(
            metadata.get("remaining_depth", max(0.0, max_depth - depth)))
        completion_ratio = min(1.0, depth / max_depth) if max_depth > 0 else 0.0

        max_tokens = self._safe_float(getattr(request, "max_tokens", 0.0))
        output_tokens = self._safe_float(getattr(request, "num_output_tokens", 0.0))
        output_near_completion = (
            output_tokens > 0 and max_tokens > 0 and
            (max_tokens - output_tokens <= 16 or output_tokens / max_tokens >= 0.85))
        graph_near_completion = (
            max_depth > 0 and remaining_depth <= self.near_completion_remaining_depth)
        near_completion = bool(
            metadata.get("near_completion", False)
            or output_near_completion
            or graph_near_completion)

        return RequestImportance(
            request_id=request_id,
            agent_type=agent_type,
            static_priority=static_priority,
            queue_priority=queue_priority,
            agent_type_score=agent_type_score,
            normalized_score=normalized_score,
            is_critical=is_critical,
            completion_ratio=completion_ratio,
            remaining_depth=remaining_depth,
            near_completion=near_completion,
        )

    def evaluate_offload(
        self,
        *,
        request_id: str,
        freed_blocks: int,
        cpu_free_blocks: int,
        predicted_duration: float,
        transfer_time: float,
        queue_fit_request: Any,
        queue_fit_demand_blocks: int,
        pressure: PressureSnapshot,
        request: Any = None,
        metadata: Optional[dict[str, Any]] = None,
        agent_scheduler: Any = None,
        priority_getter: Optional[Callable[[Any], float]] = None,
    ) -> OffloadDecision:
        metadata = dict(metadata or getattr(request, "agent_info", {}) or {})
        importance = self.get_request_importance(
            request_id,
            request=request,
            metadata=metadata,
            agent_scheduler=agent_scheduler,
            priority_getter=priority_getter,
        )
        queue_fit_request_id = (None if queue_fit_request is None else
                                getattr(queue_fit_request, "request_id", None))
        freed_blocks = max(0, int(freed_blocks))
        cpu_free_blocks = max(0, int(cpu_free_blocks))
        predicted_duration = max(0.0, float(predicted_duration or 0.0))
        transfer_time = max(0.0, float(transfer_time or 0.0))
        stall_margin = predicted_duration - transfer_time

        pressure_score = max(
            0.0,
            (pressure.gpu_usage - self.min_gpu_usage_for_offload) * 6.0,
        ) + min(2.0, pressure.waiting_demand_blocks / max(1, freed_blocks + 1))
        fit_score = min(2.0, queue_fit_demand_blocks / max(1, freed_blocks))
        stall_score = min(6.0, stall_margin / max(0.25, transfer_time))
        importance_penalty = importance.normalized_score * 0.35
        completion_penalty = 2.0 if importance.near_completion else 0.0
        upload_safety_score = 1.0 if predicted_duration >= (2.0 * transfer_time) else -1.0
        cpu_capacity_score = 1.0 if cpu_free_blocks >= freed_blocks else -4.0
        churn_penalty = self._churn_penalty(request_id)
        final_score = (stall_score + pressure_score + fit_score +
                       upload_safety_score + cpu_capacity_score -
                       importance_penalty - completion_penalty - churn_penalty)

        emergency = (pressure.gpu_usage >= self.high_pressure_gpu_usage and
                     stall_margin >= max(4.0 * transfer_time, 5.0) and
                     pressure.waiting_request_count > 0)
        low_pressure = pressure.gpu_usage < self.min_gpu_usage_for_offload
        no_queue_pressure = (
            pressure.waiting_demand_blocks <= pressure.gpu_free_blocks and
            pressure.critical_waiting_demand_blocks <=
            pressure.agent_shared_available)
        reason = "beneficial-offload"
        allowed = True
        if freed_blocks <= 0:
            allowed = False
            reason = "no-offloadable-blocks"
        elif (metadata.get("workload_profile") == "code"
              and metadata.get("offload_eligible") is not True):
            allowed = False
            reason = "paper-window-ineligible"
        elif (metadata.get("workload_profile") == "code"
              and metadata.get("reusable_prefix") is not True):
            allowed = False
            reason = "paper-prefix-not-reusable"
        elif queue_fit_request is None:
            allowed = False
            reason = "no-fitting-waiting-request"
        elif cpu_free_blocks < freed_blocks:
            allowed = False
            reason = "insufficient-cpu-blocks"
        elif predicted_duration <= transfer_time:
            allowed = False
            reason = "stall-too-short"
        elif (stall_margin < max(0.05, transfer_time * 2.0)
              and not emergency):
            allowed = False
            reason = "insufficient-stall-cover"
        elif low_pressure and no_queue_pressure:
            allowed = False
            reason = "low-pressure-enough-gpu"
        elif low_pressure and queue_fit_demand_blocks < freed_blocks:
            allowed = False
            reason = "low-pressure-poor-fit"
        elif low_pressure and pressure.waiting_request_count == 0:
            allowed = False
            reason = "low-pressure-no-backlog"
        elif (importance.normalized_score >= self.high_importance_threshold and
              importance.near_completion and not emergency):
            allowed = False
            reason = "protected-high-importance-near-completion"
        elif (importance.normalized_score >= self.high_importance_threshold and
              stall_margin < max(0.5, transfer_time * 8.0) and not emergency):
            allowed = False
            reason = "protected-high-importance-short-stall"
        elif (importance.normalized_score >= self.high_importance_threshold and
              upload_safety_score < 0 and not emergency):
            allowed = False
            reason = "protected-upload-unsafe"
        elif final_score < self.offload_score_threshold and not emergency:
            allowed = False
            reason = "score-below-threshold"

        decision = OffloadDecision(
            request_id=request_id,
            allowed=allowed,
            reason=reason,
            final_score=final_score,
            predicted_duration=predicted_duration,
            transfer_time=transfer_time,
            stall_margin=stall_margin,
            pressure_score=pressure_score,
            fit_score=fit_score,
            importance_penalty=importance_penalty,
            completion_penalty=completion_penalty,
            upload_safety_score=upload_safety_score,
            cpu_capacity_score=cpu_capacity_score,
            churn_penalty=churn_penalty,
            freed_blocks=freed_blocks,
            cpu_free_blocks=cpu_free_blocks,
            queue_fit_request_id=queue_fit_request_id,
            queue_fit_demand_blocks=queue_fit_demand_blocks,
            request_importance=importance,
            pressure=pressure,
        )
        return decision

    def record_offload_committed(self, request_id: str) -> None:
        self.recent_offload_step_by_request[request_id] = self.step_id
        self._recent_swap_steps.append(self.step_id)

    def _churn_penalty(self, request_id: str) -> float:
        penalty = 0.0
        last_offload_step = self.recent_offload_step_by_request.get(request_id)
        if (last_offload_step is not None and
                self.step_id - last_offload_step <= self.churn_cooldown_steps):
            penalty += 1.5
        if len(self._recent_swap_steps) > self.global_churn_budget:
            penalty += min(4.0, (len(self._recent_swap_steps) -
                                 self.global_churn_budget) / 8.0)
        return penalty

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
