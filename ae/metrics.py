"""Collect current vLLM counters without depending on the old MCP debug API."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import httpx
from prometheus_client.parser import text_string_to_metric_families

PREFIXES = (
    "vllm:tokencake_",
    "vllm:kv_offload_",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions",
    "vllm:prefix_cache_",
    "vllm:external_prefix_cache_",
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
    "vllm:request_success",
)


def parse_metrics(text: str) -> list[dict]:
    return [
        {"name": sample.name, "labels": sample.labels, "value": float(sample.value)}
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name.startswith(PREFIXES)
        and not sample.name.endswith(("_created", "_bucket"))
    ]


def total(samples: list[dict], name: str, **labels) -> float | None:
    values = [
        sample["value"]
        for sample in samples
        if sample["name"] == name
        and all(
            str(sample["labels"].get(key, "")).lower() == value.lower()
            if key == "transfer_type"
            else sample["labels"].get(key) == value
            for key, value in labels.items()
        )
    ]
    return sum(values) if values else None


def summarize(before: list[dict], after: list[dict], samples: list[list[dict]]) -> dict:
    def delta(name, **labels):
        end = total(after, name, **labels)
        if end is None:
            return None
        value = end - (total(before, name, **labels) or 0)
        if value < 0:
            raise ValueError(f"Counter reset during the experiment: {name}")
        return value

    def tc(group, outcome):
        return delta(f"vllm:tokencake_{group}_total", outcome=outcome)

    def maximum(name):
        values = [
            value for sample in samples if (value := total(sample, name)) is not None
        ]
        return max(values) if values else None

    scheduling = {
        key: tc("scheduling", key)
        for key in (
            "physical_preempted",
            "reservation_preempted",
            "recomputed_tokens",
            "executed_tokens",
            "resume_gpu_hit_tokens",
            "resume_cpu_hit_tokens",
            "critical_wait_ge_60s",
            "critical_wait_ge_180s",
            "reservation_denied",
            "prefill_capacity_denied",
            "generation_capacity_denied",
            "resume_deferred",
        )
    }
    decisions = {
        sample["labels"]["outcome"]: tc("decision", sample["labels"]["outcome"])
        for sample in after
        if sample["name"] == "vllm:tokencake_decision_total"
    }
    transfers = {
        direction: {
            "bytes": delta(
                "vllm:kv_offload_total_bytes_total", transfer_type=direction
            ),
            "seconds": delta(
                "vllm:kv_offload_total_time_total", transfer_type=direction
            ),
            "operations": delta("vllm:kv_offload_size_count", transfer_type=direction),
        }
        for direction in ("gpu_to_cpu", "cpu_to_gpu")
    }
    result = {
        "scheduling": scheduling,
        "decisions": decisions,
        "transfers": transfers,
        "native_preemptions": delta("vllm:num_preemptions_total"),
        "gpu_prefix_hit_tokens": delta("vllm:prefix_cache_hits_total"),
        "cpu_prefix_hit_tokens": delta("vllm:external_prefix_cache_hits_total"),
        "saved_gpu_blocks": tc("saved_blocks", "completed"),
        "critical_queue_wait_max_s": maximum(
            "vllm:tokencake_critical_queue_wait_max_seconds"
        ),
        "critical_growth_wait_max_s": maximum(
            "vllm:tokencake_critical_growth_wait_max_seconds"
        ),
        "gpu_kv_cache_peak_fraction": maximum("vllm:kv_cache_usage_perc"),
        "metric_units": {
            "transfer_seconds": "sum of operation durations; operations may overlap",
            "decisions": "counter events, including repeated evaluations",
            "missing": "null means the server did not expose this metric",
        },
    }
    return result


class MetricsCollector:
    def __init__(self, port: int, output_dir: Path):
        self.url = f"http://127.0.0.1:{port}/metrics"
        self.output_dir = output_dir
        self.samples: list[list[dict]] = []
        self.errors: list[str] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        with httpx.Client(timeout=5.0, trust_env=False) as client:
            response = client.get(self.url)
            response.raise_for_status()
        parsed = parse_metrics(response.text)
        if not parsed:
            raise ValueError("The server did not expose vLLM metrics")
        self.samples.append(parsed)
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_s": time.time(), "samples": parsed}) + "\n")

    def _run(self):
        while not self.stop_event.wait(1.0):
            try:
                self._sample()
            except Exception as error:
                self.errors.append(str(error))

    def start(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample()
        self.thread.start()

    def stop(self) -> dict:
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=10.0)
            if self.thread.is_alive():
                raise RuntimeError("Metrics collector did not terminate")
        # The server publishes asynchronous scheduler statistics once a second.
        time.sleep(1.2)
        self._sample()
        if len(self.samples) < 2:
            raise RuntimeError("Missing the initial or final metrics snapshot")
        result = summarize(self.samples[0], self.samples[-1], self.samples)
        result["sample_count"] = len(self.samples)
        result["sampling_errors"] = self.errors
        return result


def legacy_summary_fields(metrics: dict) -> dict:
    """Keep the existing case file shape, with unavailable old fields as null."""
    decisions = metrics["decisions"]
    allowed = decisions.get("selected")
    reasons = {
        key: value
        for key, value in decisions.items()
        if key
        in {
            "not_eligible",
            "empty",
            "low_pressure",
            "no_waiting_demand",
            "backoff",
            "unprofitable",
            "prefix_gap",
            "cpu_capacity",
            "stale_snapshot",
            "snapshot_external",
            "fence_wait",
        }
    }
    rejected = sum(reasons.values()) if decisions else None
    count = allowed + rejected if allowed is not None else None
    d2h = metrics["transfers"]["gpu_to_cpu"]["operations"]
    h2d = metrics["transfers"]["cpu_to_gpu"]["operations"]
    peak = metrics["gpu_kv_cache_peak_fraction"]
    return {
        "gpu_kv_cache_peak_percent": peak * 100 if peak is not None else None,
        "cpu_kv_cache_peak_percent": None,
        "cpu_total_cache_hit_count": None,
        "swap_in_count": None,
        "swap_out_count": None,
        "cpu_evict_count": None,
        "offload_events": d2h,
        "upload_events": h2d,
        "coordinator_event_counts": {},
        "coordinator_reason_counts": {
            f"offload_decision:{key}": value for key, value in reasons.items()
        },
        "offload_decision_count": count,
        "offload_allowed_count": allowed,
        "offload_rejected_count": rejected,
        "offload_reject_rate": rejected / count if count else None,
        "offload_candidate_freed_blocks": None,
        "offload_allowed_freed_blocks": None,
        "offload_committed_blocks": metrics["saved_gpu_blocks"],
        "upload_committed_blocks": None,
        "upload_reservation_count": None,
        "upload_reserved_blocks": None,
    }
