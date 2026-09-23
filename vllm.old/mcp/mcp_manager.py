import json
import heapq
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock

logger = init_logger(__name__)

DEFAULT_ESTIMATED_TIME = 1.0
DEFAULT_PRE_UPLOAD_TIME = 0.1
DEFAULT_PREDICTION_ALPHA = 0.5
DEFAULT_TRANSFER_BASE_TIME_S = 0.0
DEFAULT_TRANSFER_TIME_PER_BLOCK_S = 5e-05


def _read_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(
            "[MCPFunctionManager] invalid float override %s=%r; keep default %.6f",
            name,
            value,
            default,
        )
        return default


class MCPFunctionManager:

    def __init__(
        self,
        default_estimated_time: float = DEFAULT_ESTIMATED_TIME,
        pre_upload_time: float = DEFAULT_PRE_UPLOAD_TIME,
        prediction_alpha: float = DEFAULT_PREDICTION_ALPHA,
        transfer_base_time_s: float = DEFAULT_TRANSFER_BASE_TIME_S,
        transfer_time_per_block_s: float = DEFAULT_TRANSFER_TIME_PER_BLOCK_S,
    ):
        self.requests_to_offload: Set[str] = set()
        self.offloaded_request_data: Dict[str, List[KVCacheBlock]] = {}
        self.requests_to_upload: Set[str] = set()
        self.finished_request_blocks: Dict[str, List[KVCacheBlock]] = {}
        self.finished_request_block_hashes: Dict[str, Dict[
            int, Optional[BlockHashType]]] = {}

        self.request_prediction_key: Dict[str, str] = {}
        self.request_base_prediction_key: Dict[str, str] = {}
        self.request_metadata: Dict[str, dict[str, Any]] = {}
        self.request_start_time: Dict[str, float] = {}
        self.request_initial_estimate: Dict[str, float] = {}
        self.request_predicted_duration: Dict[str, float] = {}
        self.request_expected_finish_time: Dict[str, float] = {}
        self.duration_ewma_by_key: Dict[str, float] = {}
        self._deadline_generation: Dict[str, int] = {}
        self._upload_deadlines: list[tuple[float, int, str]] = []

        self.default_estimated_time = _read_env_float(
            "VLLM_MCP_DEFAULT_ESTIMATED_TIME_S",
            default_estimated_time,
        )
        self.pre_upload_time = _read_env_float(
            "VLLM_MCP_PRE_UPLOAD_TIME_S",
            pre_upload_time,
        )
        self.prediction_alpha = _read_env_float(
            "VLLM_MCP_PREDICTION_ALPHA",
            prediction_alpha,
        )
        self.transfer_base_time_s = _read_env_float(
            "VLLM_MCP_TRANSFER_BASE_TIME_S",
            transfer_base_time_s,
        )
        self.transfer_time_per_block_s = _read_env_float(
            "VLLM_MCP_TRANSFER_TIME_PER_BLOCK_S",
            transfer_time_per_block_s,
        )
        self.transfer_submission_time_per_run_s = _read_env_float(
            "VLLM_MCP_TRANSFER_SUBMISSION_PER_RUN_S", 3e-6)
        self.transfer_base_time_by_direction = {
            direction:
            _read_env_float(
                f"VLLM_MCP_{direction.upper()}_BASE_TIME_S",
                self.transfer_base_time_s,
            )
            for direction in ("d2h", "h2d")
        }
        self.transfer_bandwidth_bytes_per_s = {
            direction:
            max(
                1.0,
                _read_env_float(f"VLLM_MCP_{direction.upper()}_BANDWIDTH_GBPS",
                                12.0) * 1e9,
            )
            for direction in ("d2h", "h2d")
        }
        self.logical_block_bytes = 0
        self.request_transfer_states: Dict[str, str] = {}
        self._pending_request_transfers: Dict[str, List[dict[str, Any]]] = {}

        logger.info(
            "[MCPFunctionManager] config: default_estimated_time=%.6f pre_upload_time=%.6f "
            "prediction_alpha=%.6f transfer_base_time_s=%.6f "
            "transfer_time_per_block_s=%.6f",
            self.default_estimated_time,
            self.pre_upload_time,
            self.prediction_alpha,
            self.transfer_base_time_s,
            self.transfer_time_per_block_s,
        )

        self.coordination_event_counts: Dict[str, int] = {}
        self.coordination_reason_counts: Dict[str, int] = {}
        self.coordination_metric_totals: Dict[str, float] = {}
        self._candidate_request_lifecycles: Set[str] = set()
        self._committed_request_lifecycles: Set[str] = set()
        self._coordination_log_signatures: Set[tuple[str, Optional[str],
                                                     str]] = set()
        self.log_all_coordination_events = (os.getenv(
            "VLLM_MCP_LOG_ALL_COORDINATION_EVENTS", "0").lower()
                                            in {"1", "true", "yes", "on"})

    def register_request(
        self,
        request_id: str,
        *,
        prediction_key: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        base_key = prediction_key or request_id
        self.request_base_prediction_key[request_id] = base_key
        self.request_prediction_key[request_id] = base_key
        if metadata is not None:
            self.request_metadata[request_id] = dict(metadata)
        self.request_transfer_states.setdefault(request_id, "GPU_READY")

    def cancel_request_transfers(self, request_id: str) -> None:
        self._pending_request_transfers.pop(request_id, None)
        self.request_transfer_states.pop(request_id, None)

    def record_transfer_pending(
        self,
        request_id: str,
        *,
        direction: str,
        gpu_blocks: list[int],
        cpu_blocks: list[int],
    ) -> None:
        if direction not in {"d2h", "h2d"}:
            raise ValueError(f"Unknown transfer direction: {direction}")
        self.request_transfer_states[request_id] = (
            "D2H_PENDING" if direction == "d2h" else "H2D_PENDING")
        self._pending_request_transfers.setdefault(request_id, []).append({
            "direction":
            direction,
            "gpu_blocks":
            set(gpu_blocks),
            "cpu_blocks":
            set(cpu_blocks),
        })

    def record_transfer_ready(self, request_id: str, direction: str) -> None:
        self.request_transfer_states[request_id] = ("CPU_READY" if direction
                                                    == "d2h" else "GPU_READY")

    def record_transfer_completions(self, events: list[dict]) -> list[str]:
        changed_requests: list[str] = []
        for request_id, pending in list(
                self._pending_request_transfers.items()):
            remaining = []
            for transfer in pending:
                completion = next(
                    (event for event in events
                     if event.get("direction") == transfer["direction"]
                     and transfer["gpu_blocks"].issubset(
                         set(event.get("gpu_blocks", ())))
                     and transfer["cpu_blocks"].issubset(
                         set(event.get("cpu_blocks", ())))), None)
                if completion is None:
                    remaining.append(transfer)
                    continue
                direction = transfer["direction"]
                if completion.get("failed"):
                    self.request_transfer_states[request_id] = (
                        "GPU_READY" if direction == "d2h" else "CPU_READY")
                else:
                    self.record_transfer_ready(request_id, direction)
                if request_id not in changed_requests:
                    changed_requests.append(request_id)
            if remaining:
                self._pending_request_transfers[request_id] = remaining
            else:
                self._pending_request_transfers.pop(request_id, None)
        return changed_requests

    def has_requests(self) -> bool:
        return (len(self.requests_to_offload) > 0
                or len(self.requests_to_upload) > 0)

    def has_scheduler_work(self,
                           *,
                           has_waiting: bool,
                           now: Optional[float] = None) -> bool:
        if self.requests_to_upload:
            return True
        if has_waiting and self.requests_to_offload:
            return True
        if not self.offloaded_request_data:
            return False

        self._schedule_untracked_deadlines()
        now = time.time() if now is None else now
        return bool(self._upload_deadlines
                    and self._upload_deadlines[0][0] <= now)

    def add_request_to_offload(self,
                               request_id: str,
                               estimated_time: Optional[float] = None,
                               request_type: Optional[str] = None,
                               *,
                               started_at: Optional[float] = None):
        if request_type:
            metadata = self.request_metadata.setdefault(request_id, {})
            metadata["request_type"] = request_type
            base_key = self.request_base_prediction_key.get(
                request_id,
                self.request_prediction_key.get(request_id, request_id),
            )
            self.request_prediction_key[request_id] = (
                f"{base_key}|{request_type}")
        self.requests_to_offload.add(request_id)
        now = time.time()
        if started_at is not None:
            try:
                started_at = min(now, float(started_at))
            except (TypeError, ValueError):
                started_at = None
        stall_start_time = now if started_at is None else started_at
        predicted_duration = self.predict_duration(request_id, estimated_time)
        self.request_start_time[request_id] = stall_start_time
        self.request_initial_estimate[request_id] = (
            self.default_estimated_time
            if estimated_time is None else estimated_time)
        self.request_predicted_duration[request_id] = predicted_duration
        self.request_expected_finish_time[request_id] = (stall_start_time +
                                                         predicted_duration)
        logger.debug(
            "[MCPFunctionManager] add request to offload: %s predicted_duration=%.4f pending=%s",
            request_id,
            predicted_duration,
            len(self.requests_to_offload),
        )

    def get_requests_to_offload(self) -> Set[str]:
        return self.requests_to_offload.copy()

    def mark_request_offloaded(self, request_id: str,
                               cpu_blocks: List[KVCacheBlock]):
        if request_id in self.requests_to_offload:
            logger.info(
                "[MCPFunctionManager] mark request offloaded: %s, cpu_blocks: %s",
                request_id, len(cpu_blocks))
            self.requests_to_offload.remove(request_id)
            self.offloaded_request_data[request_id] = cpu_blocks
            self._schedule_request_deadlines(request_id)

    def get_offloaded_request_data(self,
                                   request_id: str) -> List[KVCacheBlock]:
        return self.offloaded_request_data.get(request_id, [])

    def mark_request_finished(self,
                              request_id: str,
                              observed_duration: Optional[float] = None,
                              request_type: Optional[str] = None):
        if request_type:
            self.request_metadata.setdefault(request_id,
                                             {})["request_type"] = request_type
        if observed_duration is None and request_id in self.request_start_time:
            observed_duration = max(
                0.0,
                time.time() - self.request_start_time[request_id])
        self._update_duration_history(request_id, observed_duration)

        was_offloaded = request_id in self.offloaded_request_data
        if was_offloaded:
            self.requests_to_upload.add(request_id)
            logger.info(
                "[MCPFunctionManager] mark offloaded request finished: %s",
                request_id)
        elif request_id in self.requests_to_offload:
            self.requests_to_offload.remove(request_id)
            logger.debug(
                "[MCPFunctionManager] mark pending offload request finished: %s",
                request_id,
            )
        else:
            logger.debug(
                "[MCPFunctionManager] ignore finished signal for untracked request %s",
                request_id,
            )

        self.request_expected_finish_time.pop(request_id, None)
        self.request_predicted_duration.pop(request_id, None)
        self._invalidate_request_deadlines(request_id)
        if not was_offloaded:
            self._candidate_request_lifecycles.discard(request_id)
            self._committed_request_lifecycles.discard(request_id)

    def _predictive_upload_lead_time(self, request_id: str) -> float:
        block_count = len(self.offloaded_request_data.get(request_id, []))
        transfer_time = self.estimate_transfer_time(block_count,
                                                    direction="h2d")
        return max(self.pre_upload_time, transfer_time * 1.25)

    def _invalidate_request_deadlines(self, request_id: str) -> int:
        generation = self._deadline_generation.get(request_id, 0) + 1
        self._deadline_generation[request_id] = generation
        return generation

    def _schedule_request_deadlines(self, request_id: str) -> None:
        expected_finish_time = self.request_expected_finish_time.get(
            request_id)
        if expected_finish_time is None:
            return
        generation = self._invalidate_request_deadlines(request_id)
        upload_lead_time = self._predictive_upload_lead_time(request_id)
        heapq.heappush(
            self._upload_deadlines,
            (expected_finish_time - upload_lead_time, generation, request_id),
        )

    def _schedule_untracked_deadlines(self) -> None:
        for request_id in self.offloaded_request_data:
            if request_id not in self._deadline_generation:
                self._schedule_request_deadlines(request_id)

    def _pop_due_deadlines(
        self,
        heap: list[tuple[float, int, str]],
        *,
        now: float,
    ) -> set[str]:
        due: set[str] = set()
        while heap and heap[0][0] <= now:
            _, generation, request_id = heapq.heappop(heap)
            if self._deadline_generation.get(request_id) != generation:
                continue
            if request_id not in self.offloaded_request_data:
                continue
            due.add(request_id)
        return due

    def get_requests_to_upload(self) -> Set[str]:
        self._schedule_untracked_deadlines()
        self.requests_to_upload.update(
            self._pop_due_deadlines(self._upload_deadlines, now=time.time()))

        return self.requests_to_upload.copy()

    def mark_request_uploaded(self, request_id: str):
        if request_id in self.requests_to_upload:
            logger.info("[MCPFunctionManager] mark request uploaded: %s",
                        request_id)
            self.requests_to_upload.remove(request_id)

        self.offloaded_request_data.pop(request_id, None)
        self.finished_request_blocks.pop(request_id, None)
        self.finished_request_block_hashes.pop(request_id, None)
        self.request_start_time.pop(request_id, None)
        self.request_initial_estimate.pop(request_id, None)
        self.request_predicted_duration.pop(request_id, None)
        self.request_expected_finish_time.pop(request_id, None)
        self.request_metadata.pop(request_id, None)
        self.request_prediction_key.pop(request_id, None)
        self.request_base_prediction_key.pop(request_id, None)
        self._candidate_request_lifecycles.discard(request_id)
        self._committed_request_lifecycles.discard(request_id)
        self._invalidate_request_deadlines(request_id)

    def save_finished_request_blocks(
        self,
        request_id: str,
        blocks: List[KVCacheBlock],
    ) -> None:
        logger.debug(
            "[MCPFunctionManager] save finished request blocks: %s, %s",
            request_id, len(blocks))
        self.finished_request_blocks[request_id] = blocks
        self.finished_request_block_hashes[request_id] = {
            block.block_id: block.block_hash
            for block in blocks
        }

    def get_finished_request_blocks(self,
                                    request_id: str) -> List[KVCacheBlock]:
        saved_hashes = self.finished_request_block_hashes.get(request_id, {})
        return [
            block
            for block in self.finished_request_blocks.get(request_id, [])
            if block.block_hash is not None
            and block.block_hash == saved_hashes.get(block.block_id)
        ]

    def predict_duration(self,
                         request_id: str,
                         estimated_time: Optional[float] = None) -> float:
        key = self.request_prediction_key.get(request_id, request_id)
        historical = self.duration_ewma_by_key.get(key)

        if estimated_time is None and historical is None:
            return self.default_estimated_time
        if estimated_time is None:
            return historical
        if historical is None:
            return estimated_time
        return max(0.0, 0.5 * estimated_time + 0.5 * historical)

    def get_predicted_duration(self, request_id: str) -> float:
        if request_id in self.request_predicted_duration:
            return self.request_predicted_duration[request_id]
        return self.predict_duration(request_id)

    def get_remaining_predicted_duration(
        self,
        request_id: str,
        *,
        now: Optional[float] = None,
    ) -> float:
        expected_finish_time = self.request_expected_finish_time.get(
            request_id)
        if expected_finish_time is None:
            return self.get_predicted_duration(request_id)
        now = time.time() if now is None else now
        return max(0.0, expected_finish_time - now)

    def configure_transfer_layout(self, logical_block_bytes: int) -> None:
        self.logical_block_bytes = max(0, int(logical_block_bytes))

    def estimate_transfer_time(
        self,
        num_blocks: int,
        *,
        direction: str = "roundtrip",
        copy_runs: Optional[int] = None,
    ) -> float:
        num_blocks = max(0, int(num_blocks))
        copy_runs = num_blocks if copy_runs is None else max(0, int(copy_runs))
        if direction == "roundtrip":
            return sum(
                self.estimate_transfer_time(
                    num_blocks, direction=copy_direction, copy_runs=copy_runs)
                for copy_direction in ("d2h", "h2d"))
        if direction not in {"d2h", "h2d"}:
            raise ValueError(f"Unknown transfer direction: {direction}")
        bandwidth = self.transfer_bandwidth_bytes_per_s[direction]
        if self.logical_block_bytes:
            payload_time = num_blocks * self.logical_block_bytes / bandwidth
        else:
            payload_time = self.transfer_time_per_block_s * num_blocks
        return (self.transfer_base_time_by_direction[direction] +
                payload_time +
                self.transfer_submission_time_per_run_s * copy_runs)

    def increment_coordination_counter(self,
                                       event: str,
                                       *,
                                       reason: str = "") -> None:
        self.coordination_event_counts[event] = (
            self.coordination_event_counts.get(event, 0) + 1)
        if reason:
            key = f"{event}:{reason}"
            self.coordination_reason_counts[key] = (
                self.coordination_reason_counts.get(key, 0) + 1)

    def record_offload_decision(
        self,
        request_id: str,
        *,
        allowed: bool,
        reason: str,
        predicted_duration: float,
        transfer_time: float,
        queue_fit_request_id: Optional[str],
        freed_blocks: int,
        gpu_usage: float,
        **details: Any,
    ) -> None:
        decision = {
            "allowed": allowed,
            "reason": reason,
            "predicted_duration": predicted_duration,
            "transfer_time": transfer_time,
            "queue_fit_request_id": queue_fit_request_id,
            "freed_blocks": freed_blocks,
            "gpu_usage": gpu_usage,
        }
        decision.update(details)
        eligible_blocks = int(details.get("eligible_suffix_blocks", 0) or 0)
        if (eligible_blocks > 0
                and request_id not in self._candidate_request_lifecycles):
            self._candidate_request_lifecycles.add(request_id)
            self._add_coordination_metric("valid_offload_candidate_requests",
                                          1)
        self.record_coordination_event("offload_decision", request_id,
                                       **decision)

    def record_coordination_event(
        self,
        event: str,
        request_id: Optional[str] = None,
        **fields: Any,
    ) -> None:
        payload = {"event": event}
        if request_id is not None:
            payload["request_id"] = request_id
        payload.update(fields)
        reason = str(payload.get("reason", ""))
        self.coordination_event_counts[event] = (
            self.coordination_event_counts.get(event, 0) + 1)
        if reason:
            reason_key = f"{event}:{reason}"
            self.coordination_reason_counts[reason_key] = (
                self.coordination_reason_counts.get(reason_key, 0) + 1)
        self._record_coordination_metric_totals(event, payload)
        debug_only = event == "pressure_snapshot"
        log_fn = logger.debug if debug_only else logger.info
        if (not self.log_all_coordination_events
                and event == "offload_decision"
                and payload.get("allowed") is False):
            signature = (event, request_id, reason)
            if signature in self._coordination_log_signatures:
                log_fn = logger.debug
                debug_only = True
            else:
                self._coordination_log_signatures.add(signature)
        if debug_only and not logger.isEnabledFor(logging.DEBUG):
            return
        try:
            log_fn("[AgentOffloadCoordinator] %s %s", event,
                   json.dumps(payload, sort_keys=True, default=str))
        except TypeError:
            log_fn("[AgentOffloadCoordinator] %s %s", event, payload)

    def get_debug_state(self) -> dict[str, Any]:
        return {
            "coordination_event_counts":
            dict(self.coordination_event_counts),
            "coordination_reason_counts":
            dict(self.coordination_reason_counts),
            "coordination_metric_totals":
            dict(self.coordination_metric_totals),
        }

    def _add_coordination_metric(self, key: str, value: float) -> None:
        self.coordination_metric_totals[key] = (
            self.coordination_metric_totals.get(key, 0.0) + value)

    def _record_coordination_metric_totals(
        self,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        if event == "offload_decision":
            self._add_coordination_metric("offload_decision_count", 1)
            freed_blocks = float(payload.get("freed_blocks", 0) or 0)
            self._add_coordination_metric("offload_candidate_freed_blocks",
                                          freed_blocks)
            allowed = payload.get("allowed")
            if isinstance(allowed, str):
                allowed = allowed.lower() == "true"
            if allowed is None:
                allowed = payload.get("reason") == "beneficial-offload"
            if allowed:
                self._add_coordination_metric("offload_allowed_count", 1)
                self._add_coordination_metric("offload_allowed_freed_blocks",
                                              freed_blocks)
            else:
                self._add_coordination_metric("offload_rejected_count", 1)
        elif event == "offload_committed":
            raw_request_id = payload.get("request_id")
            request_id = (None
                          if raw_request_id is None else str(raw_request_id))
            if (request_id is not None
                    and request_id not in self._committed_request_lifecycles):
                self._committed_request_lifecycles.add(request_id)
                self._add_coordination_metric("committed_offload_requests", 1)
            self._add_coordination_metric(
                "offload_committed_blocks",
                float(payload.get("offloaded_blocks", 0) or 0),
            )
            self._add_coordination_metric(
                "d2h_logical_blocks",
                float(payload.get("copied_blocks", 0) or 0),
            )
            self._add_coordination_metric(
                "cpu_shadow_hit_blocks",
                float(payload.get("shadow_hit_blocks", 0) or 0),
            )
        elif event == "upload_committed":
            self._add_coordination_metric(
                "upload_committed_blocks",
                float(payload.get("uploaded_blocks", 0) or 0),
            )
            fresh_blocks = float(payload.get("fresh_blocks", 0) or 0)
            uploaded_blocks = float(payload.get("uploaded_blocks", 0) or 0)
            self._add_coordination_metric("h2d_logical_blocks", fresh_blocks)
            self._add_coordination_metric(
                "no_copy_restore_blocks",
                max(0.0, uploaded_blocks - fresh_blocks),
            )
        elif event == "pressure_snapshot":
            self._add_coordination_metric("pressure_snapshot_count", 1)
            waiting_demand = float(
                payload.get("waiting_demand_blocks", 0) or 0)
            clean_free = float(payload.get("gpu_clean_free_blocks", 0) or 0)
            self._add_coordination_metric("waiting_demand_blocks_total",
                                          waiting_demand)
            self._add_coordination_metric("gpu_clean_free_blocks_total",
                                          clean_free)
            if waiting_demand > clean_free:
                self._add_coordination_metric(
                    "pressure_waiting_exceeds_free_steps", 1)

    def _update_duration_history(self, request_id: str,
                                 observed_duration: Optional[float]) -> None:
        if observed_duration is None:
            return
        key = self.request_prediction_key.get(request_id, request_id)
        previous = self.duration_ewma_by_key.get(key)
        if previous is None:
            updated = observed_duration
        else:
            updated = ((self.prediction_alpha * observed_duration) +
                       ((1.0 - self.prediction_alpha) * previous))
        self.duration_ewma_by_key[key] = updated
