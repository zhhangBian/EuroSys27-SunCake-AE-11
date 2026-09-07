from __future__ import annotations

import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import traceback
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
from ae.metrics import MetricsCollector, legacy_summary_fields
from ae.server import build_command as build_server_command

ROOT = Path(__file__).resolve().parents[1]
CONDA_SH = Path(
    os.environ.get("CONDA_SH", Path.home() / "miniconda3/etc/profile.d/conda.sh")
)
MOONCAKE_REPO = Path(
    os.environ.get("MOONCAKE_REPO", ROOT.parent / "Mooncake-TokenCake")
).resolve()
PARROT_REPO = Path(
    os.environ.get("PARROT_REPO", ROOT.parent / "Parrot-TokenCake")
).resolve()

DEFAULT_TASKS = ["code", "research"]
DEFAULT_QPS_LIST = [0.05, 0.1, 0.2, 0.5, 1.0]
DEFAULT_NUM_LIST = [20, 30]
DEFAULT_SEED = 42
DEFAULT_MOONCAKE_GLOBAL_SEGMENT_SIZE = 3200 * 1024 * 1024
DEFAULT_MOONCAKE_LOCAL_BUFFER_SIZE = 512 * 1024 * 1024

GPU_USAGE_RE = re.compile(r"GPU KV cache usage: (?P<value>[0-9.]+)%")
CPU_STATS_RE = re.compile(
    r"CPU KV cache usage: (?P<cpu_usage>[0-9.]+)%, .*?"
    r"CPU total cache hit count: (?P<cpu_hits>\d+), .*?"
    r"swap in count: (?P<swap_in>\d+), "
    r"swap out count: (?P<swap_out>\d+), "
    r"Cpu KV cache evict count: (?P<cpu_evict>\d+)"
)
COORDINATOR_EVENT_RE = re.compile(
    r"\[AgentOffloadCoordinator\] (?P<event>[a-z_]+) (?P<payload>\{.*\})"
)
MOONCAKE_STORE_STATS_RE = re.compile(r"\[MooncakeStoreStats\] (?P<payload>\{.*\})")
MOONCAKE_CONNECTOR_STATS_RE = re.compile(
    r"\[MooncakeConnectorStats\] (?P<payload>\{.*\})"
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str


@dataclass(frozen=True)
class ModeSpec:
    name: str
    runner_kind: str = "vllm"
    offloading_enabled: bool = False
    scheduling_policy: str = "fcfs"
    enable_agent_scheduling: bool = False
    extra_server_args: str = ""
    extra_client_args: tuple[str, ...] = ()
    extra_env: tuple[tuple[str, str], ...] = ()
    gpu_memory_utilization: Optional[float] = None
    trust_remote_code: bool = False


@dataclass
class CaseSummary:
    experiment: str
    mode: str
    task: str
    model_name: str
    model_path: str
    request_rate: float
    num_requests: int
    total_runtime_s: float
    throughput_rps: float
    avg_app_latency_s: float
    p50_app_latency_s: float
    p90_app_latency_s: float
    p95_app_latency_s: float
    p99_app_latency_s: float
    max_app_latency_s: float
    avg_llm_latency_s: float
    avg_tool_latency_s: float
    avg_residual_latency_s: float
    gpu_kv_cache_peak_percent: Optional[float]
    cpu_kv_cache_peak_percent: Optional[float]
    cpu_total_cache_hit_count: Optional[int]
    swap_in_count: Optional[int]
    swap_out_count: Optional[int]
    cpu_evict_count: Optional[int]
    offload_events: int
    upload_events: int
    coordinator_event_counts: dict[str, int]
    coordinator_reason_counts: dict[str, int]
    client_log_path: str
    server_log_path: str
    app_json_path: str
    command_path: str
    offload_decision_count: int = 0
    offload_allowed_count: int = 0
    offload_rejected_count: int = 0
    offload_reject_rate: Optional[float] = None
    offload_candidate_freed_blocks: int = 0
    offload_allowed_freed_blocks: int = 0
    offload_committed_blocks: int = 0
    upload_committed_blocks: int = 0
    upload_reservation_count: int = 0
    upload_reserved_blocks: int = 0
    mooncake_store_put_count: int = 0
    mooncake_store_get_count: int = 0
    mooncake_store_get_hit_count: int = 0
    mooncake_store_get_miss_count: int = 0
    mooncake_store_put_bytes: int = 0
    mooncake_store_get_bytes: int = 0
    mooncake_store_put_ms: float = 0.0
    mooncake_store_get_ms: float = 0.0
    mooncake_connector_send_count: int = 0
    mooncake_connector_recv_count: int = 0
    mooncake_connector_recv_hit_count: int = 0
    mooncake_connector_recv_miss_count: int = 0
    mooncake_connector_bypass_count: int = 0
    mooncake_connector_recv_batch_count: int = 0
    mooncake_connector_send_ms: float = 0.0
    mooncake_connector_recv_batch_ms: float = 0.0
    mooncake_error_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime_metrics: dict[str, Any] = field(default_factory=dict)
    dag_e2e_s: float = 0.0


@dataclass
class BlockedSummary:
    experiment: str
    mode: str
    reason: str
    details: dict[str, Any]


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def get_model_specs() -> dict[str, ModelSpec]:
    return {
        "qwen2.5-7b-instruct": ModelSpec(
            name="qwen2.5-7b-instruct",
            path=os.environ.get("SUNCAKE_MODEL_7B", "Qwen/Qwen2.5-7B-Instruct"),
        ),
        "qwen2.5-14b-instruct": ModelSpec(
            name="qwen2.5-14b-instruct",
            path=os.environ.get("SUNCAKE_MODEL_14B", "Qwen/Qwen2.5-14B-Instruct"),
        ),
    }


def default_mode_specs() -> dict[str, ModeSpec]:
    return {
        "vllm_vanilla": ModeSpec(
            name="vllm_vanilla", extra_server_args="--no-enable-prefix-caching"
        ),
        "baseline": ModeSpec(name="baseline"),
        "agent": ModeSpec(
            name="agent", scheduling_policy="agent", enable_agent_scheduling=True
        ),
        "offload": ModeSpec(name="offload", offloading_enabled=True),
        "offload_agent": ModeSpec(
            name="offload_agent",
            offloading_enabled=True,
            scheduling_policy="agent",
            enable_agent_scheduling=True,
        ),
        "mooncake": ModeSpec(
            name="mooncake",
            runner_kind="mooncake",
            trust_remote_code=True,
        ),
    }


def choose_free_port(preferred_port: int) -> int:
    for port in range(preferred_port, preferred_port + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find a free port starting from {preferred_port}")


def choose_mooncake_master_port(server_port: int) -> int:
    preferred_port = server_port + 10000
    if not 1024 <= preferred_port <= 65335:
        preferred_port = server_port - 10000
    if not 1024 <= preferred_port <= 65335:
        preferred_port = 50123
    return choose_free_port(preferred_port)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_server_slice(log_text: str) -> dict[str, Any]:
    gpu_peaks = [
        float(match.group("value")) for match in GPU_USAGE_RE.finditer(log_text)
    ]
    cpu_matches = list(CPU_STATS_RE.finditer(log_text))
    cpu_peak = None
    cpu_hits = None
    swap_in = None
    swap_out = None
    cpu_evict = None
    if cpu_matches:
        cpu_peak = max(float(match.group("cpu_usage")) for match in cpu_matches)
        last = cpu_matches[-1]
        cpu_hits = int(last.group("cpu_hits"))
        swap_in = int(last.group("swap_in"))
        swap_out = int(last.group("swap_out"))
        cpu_evict = int(last.group("cpu_evict"))

    coordinator_event_counts: dict[str, int] = {}
    coordinator_reason_counts: dict[str, int] = {}
    offload_decision_count = 0
    offload_allowed_count = 0
    offload_rejected_count = 0
    offload_candidate_freed_blocks = 0
    offload_allowed_freed_blocks = 0
    offload_committed_blocks = 0
    upload_committed_blocks = 0
    upload_reservation_count = 0
    upload_reserved_blocks = 0
    for match in COORDINATOR_EVENT_RE.finditer(log_text):
        event = match.group("event")
        coordinator_event_counts[event] = coordinator_event_counts.get(event, 0) + 1
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError:
            continue
        reason = payload.get("reason")
        if reason:
            key = f"{event}:{reason}"
            coordinator_reason_counts[key] = coordinator_reason_counts.get(key, 0) + 1
        if event == "offload_decision":
            offload_decision_count += 1
            freed_blocks = int(payload.get("freed_blocks", 0) or 0)
            offload_candidate_freed_blocks += freed_blocks
            allowed = payload.get("allowed")
            if isinstance(allowed, str):
                allowed = allowed.lower() == "true"
            if allowed is None:
                allowed = reason == "beneficial-offload"
            if allowed:
                offload_allowed_count += 1
                offload_allowed_freed_blocks += freed_blocks
            else:
                offload_rejected_count += 1
        elif event == "offload_committed":
            offload_committed_blocks += int(payload.get("offloaded_blocks", 0) or 0)
        elif event == "upload_committed":
            upload_committed_blocks += int(payload.get("uploaded_blocks", 0) or 0)
        elif event == "upload_reservation":
            upload_reservation_count += 1
            upload_reserved_blocks += int(
                payload.get("newly_reserved", payload.get("reserve_now", 0)) or 0
            )

    offload_reject_rate = (
        offload_rejected_count / offload_decision_count
        if offload_decision_count > 0
        else None
    )

    mooncake_store_put_count = 0
    mooncake_store_get_count = 0
    mooncake_store_get_hit_count = 0
    mooncake_store_get_miss_count = 0
    mooncake_store_put_bytes = 0
    mooncake_store_get_bytes = 0
    mooncake_store_put_ms = 0.0
    mooncake_store_get_ms = 0.0
    for match in MOONCAKE_STORE_STATS_RE.finditer(log_text):
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError:
            continue
        op = payload.get("op")
        elapsed_ms = float(payload.get("elapsed_ms", 0.0) or 0.0)
        byte_count = int(payload.get("bytes", 0) or 0)
        if op == "put":
            mooncake_store_put_count += 1
            mooncake_store_put_bytes += byte_count
            mooncake_store_put_ms += elapsed_ms
        elif op == "get":
            mooncake_store_get_count += 1
            mooncake_store_get_bytes += byte_count
            mooncake_store_get_ms += elapsed_ms
            if payload.get("hit"):
                mooncake_store_get_hit_count += 1
            else:
                mooncake_store_get_miss_count += 1

    mooncake_connector_send_count = 0
    mooncake_connector_recv_count = 0
    mooncake_connector_recv_hit_count = 0
    mooncake_connector_recv_miss_count = 0
    mooncake_connector_bypass_count = 0
    mooncake_connector_recv_batch_count = 0
    mooncake_connector_send_ms = 0.0
    mooncake_connector_recv_batch_ms = 0.0
    for match in MOONCAKE_CONNECTOR_STATS_RE.finditer(log_text):
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError:
            continue
        op = payload.get("op")
        if op == "send":
            mooncake_connector_send_count += 1
            mooncake_connector_send_ms += float(payload.get("elapsed_ms", 0.0) or 0.0)
        elif op == "v1_save_layer":
            mooncake_connector_send_count += 1
            mooncake_connector_send_ms += float(payload.get("elapsed_ms", 0.0) or 0.0)
        elif op == "recv":
            mooncake_connector_recv_count += 1
            if payload.get("hit"):
                mooncake_connector_recv_hit_count += 1
            else:
                mooncake_connector_recv_miss_count += 1
        elif op == "v1_load_layer":
            mooncake_connector_recv_count += 1
            if payload.get("hit"):
                mooncake_connector_recv_hit_count += 1
            else:
                mooncake_connector_recv_miss_count += 1
        elif op == "recv_batch":
            mooncake_connector_recv_batch_count += 1
            mooncake_connector_recv_batch_ms += float(
                payload.get("elapsed_ms", 0.0) or 0.0
            )
            if payload.get("bypass_model_exec"):
                mooncake_connector_bypass_count += 1
        elif op == "v1_load_request":
            mooncake_connector_recv_batch_count += 1
            mooncake_connector_recv_batch_ms += float(
                payload.get("elapsed_ms", 0.0) or 0.0
            )

    return {
        "gpu_kv_cache_peak_percent": max(gpu_peaks) if gpu_peaks else None,
        "cpu_kv_cache_peak_percent": cpu_peak,
        "cpu_total_cache_hit_count": cpu_hits,
        "swap_in_count": swap_in,
        "swap_out_count": swap_out,
        "cpu_evict_count": cpu_evict,
        "offload_events": log_text.count("mark request offloaded:"),
        "upload_events": log_text.count("mark request uploaded:"),
        "mooncake_store_put_count": mooncake_store_put_count,
        "mooncake_store_get_count": mooncake_store_get_count,
        "mooncake_store_get_hit_count": mooncake_store_get_hit_count,
        "mooncake_store_get_miss_count": mooncake_store_get_miss_count,
        "mooncake_store_put_bytes": mooncake_store_put_bytes,
        "mooncake_store_get_bytes": mooncake_store_get_bytes,
        "mooncake_store_put_ms": mooncake_store_put_ms,
        "mooncake_store_get_ms": mooncake_store_get_ms,
        "mooncake_connector_send_count": mooncake_connector_send_count,
        "mooncake_connector_recv_count": mooncake_connector_recv_count,
        "mooncake_connector_recv_hit_count": mooncake_connector_recv_hit_count,
        "mooncake_connector_recv_miss_count": mooncake_connector_recv_miss_count,
        "mooncake_connector_bypass_count": mooncake_connector_bypass_count,
        "mooncake_connector_recv_batch_count": mooncake_connector_recv_batch_count,
        "mooncake_connector_send_ms": mooncake_connector_send_ms,
        "mooncake_connector_recv_batch_ms": mooncake_connector_recv_batch_ms,
        "mooncake_error_count": (
            log_text.count("NO_AVAILABLE_HANDLE") + log_text.count("allocation_failed")
        ),
        "coordinator_event_counts": coordinator_event_counts,
        "coordinator_reason_counts": coordinator_reason_counts,
        "offload_decision_count": offload_decision_count,
        "offload_allowed_count": offload_allowed_count,
        "offload_rejected_count": offload_rejected_count,
        "offload_reject_rate": offload_reject_rate,
        "offload_candidate_freed_blocks": offload_candidate_freed_blocks,
        "offload_allowed_freed_blocks": offload_allowed_freed_blocks,
        "offload_committed_blocks": offload_committed_blocks,
        "upload_committed_blocks": upload_committed_blocks,
        "upload_reservation_count": upload_reservation_count,
        "upload_reserved_blocks": upload_reserved_blocks,
    }


def load_app_info(app_json_path: Path) -> dict[str, Any]:
    return json.loads(app_json_path.read_text(encoding="utf-8"))


def load_app_metrics(
    app_json_path: Path, expected_requests: Optional[int] = None
) -> dict[str, float]:
    app_info = load_app_info(app_json_path)
    if expected_requests is not None and len(app_info) != expected_requests:
        raise RuntimeError(
            f"Expected {expected_requests} completed applications, "
            f"got {len(app_info)} in {app_json_path}"
        )
    latencies = np.array(
        [info["app_latency"] for info in app_info.values()], dtype=float
    )
    if latencies.size == 0:
        raise RuntimeError(f"No application latencies found in {app_json_path}")
    if not np.all(np.isfinite(latencies) & (latencies > 0)):
        raise RuntimeError(f"Invalid application latencies in {app_json_path}")

    llm_latencies = []
    tool_latencies = []
    residual_latencies = []
    for app in app_info.values():
        if not app.get("app_finished", False):
            raise RuntimeError(f"Incomplete application in {app_json_path}")
        request_info = app.get("request_info", {})
        llm_total = 0.0
        tool_total = 0.0
        residual_total = 0.0
        for node_info in request_info.values():
            llm_total += float(node_info.get("llm_latency", 0.0) or 0.0)
            tool_total += float(node_info.get("tool_latency", 0.0) or 0.0)
            residual_total += float(node_info.get("residual_latency", 0.0) or 0.0)
        llm_latencies.append(llm_total)
        tool_latencies.append(tool_total)
        residual_latencies.append(
            max(0.0, app.get("app_latency", 0.0) - llm_total - tool_total)
            if residual_total == 0.0
            else residual_total
        )

    return {
        "dag_e2e_s": max(app["app_end_time"] for app in app_info.values())
        - min(
            app["app_start_time"] - app.get("arrival_offset_s", 0.0)
            for app in app_info.values()
        ),
        "avg_app_latency_s": float(np.mean(latencies)),
        "p50_app_latency_s": float(np.percentile(latencies, 50)),
        "p90_app_latency_s": float(np.percentile(latencies, 90)),
        "p95_app_latency_s": float(np.percentile(latencies, 95)),
        "p99_app_latency_s": float(np.percentile(latencies, 99)),
        "max_app_latency_s": float(np.max(latencies)),
        "avg_llm_latency_s": float(np.mean(llm_latencies)),
        "avg_tool_latency_s": float(np.mean(tool_latencies)),
        "avg_residual_latency_s": float(np.mean(residual_latencies)),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_mooncake_runtime_config(
    path: Path,
    *,
    global_segment_size: int = DEFAULT_MOONCAKE_GLOBAL_SEGMENT_SIZE,
    local_buffer_size: int = DEFAULT_MOONCAKE_LOCAL_BUFFER_SIZE,
    master_server_address: str = "127.0.0.1:50123",
) -> Path:
    base_config_path = MOONCAKE_REPO / "server_mooncake.json"
    config = json.loads(base_config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "global_segment_size": int(global_segment_size),
            "local_buffer_size": int(local_buffer_size),
            "master_server_address": master_server_address,
        }
    )
    write_json(path, config)
    return path


class BaseServerRunner:
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


def terminate_process_group(
    process: Optional[subprocess.Popen[str]], timeout_s: int = 30
) -> None:
    if process is None:
        return
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is None:
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=10)

    # The shell wrapper can exit before uvicorn/engine subprocesses finish
    # handling SIGTERM. Make a best-effort pass over the whole process group.
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return


class VllmServerRunner(BaseServerRunner):
    def __init__(
        self,
        *,
        mode: ModeSpec,
        model: ModelSpec,
        output_dir: Path,
        port: int,
        cuda_devices: str,
        gpu_memory_utilization: float,
        swap_space: float,
        max_model_len: int,
        extra_server_args: str = "",
        num_gpu_blocks_override: Optional[int] = None,
        scheduler_cls: str = "vllm.v1.core.sched.scheduler.Scheduler",
    ) -> None:
        self.mode = mode
        self.model = model
        self.output_dir = output_dir
        self.port = port
        self.cuda_devices = cuda_devices
        self.gpu_memory_utilization = (
            mode.gpu_memory_utilization
            if mode.gpu_memory_utilization is not None
            else gpu_memory_utilization
        )
        self.swap_space = swap_space
        self.max_model_len = max_model_len
        self.extra_server_args = extra_server_args
        self.num_gpu_blocks_override = num_gpu_blocks_override
        self.scheduler_cls = scheduler_cls
        self.process: Optional[subprocess.Popen[str]] = None
        self.log_handle = None
        self.log_path = output_dir / f"{mode.name}_server.log"
        self.command_repr = ""

    def build_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        for key in list(env):
            if key.startswith(("VLLM_TEST_", "VLLM_AGENT_", "VLLM_MCP_")) or key in {
                "VLLM_TEMPORAL_SELECTION_POLICY",
                "SUNCAKE_TRANSFER_BANDWIDTH_GBPS",
            }:
                del env[key]
        env["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        )
        env["VLLM_TEST_MODEL_PATH"] = self.model.path
        env["VLLM_TEST_PORT"] = str(self.port)
        env["VLLM_TEST_CUDA_VISIBLE_DEVICES"] = self.cuda_devices
        env["VLLM_TEST_GPU_MEMORY_UTILIZATION"] = str(self.gpu_memory_utilization)
        env["VLLM_TEST_SWAP_SPACE"] = str(self.swap_space)
        env["VLLM_TEST_MAX_MODEL_LEN"] = str(self.max_model_len)
        env["VLLM_TEST_SCHEDULING_POLICY"] = self.mode.scheduling_policy
        env["VLLM_TEST_ENABLE_AGENT_SCHEDULING"] = (
            "1" if self.mode.enable_agent_scheduling else "0"
        )
        env["VLLM_TEST_ENABLE_KVCACHE_CPU_OFFLOADING"] = (
            "1" if self.mode.offloading_enabled else "0"
        )
        env["VLLM_TEST_SCHEDULER_CLS"] = self.scheduler_cls
        env["SUNCAKE_PYTHON"] = sys.executable
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_devices
        env["VLLM_USE_SIMPLE_KV_OFFLOAD"] = "0"
        env["VLLM_LOG_STATS_INTERVAL"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        if self.num_gpu_blocks_override is not None:
            env["VLLM_TEST_NUM_GPU_BLOCKS_OVERRIDE"] = str(self.num_gpu_blocks_override)
        if self.mode.extra_server_args:
            env["VLLM_TEST_EXTRA_ARGS"] = self.mode.extra_server_args
        if self.extra_server_args:
            combined = " ".join(
                item
                for item in [
                    env.get("VLLM_TEST_EXTRA_ARGS", ""),
                    self.extra_server_args,
                ]
                if item
            )
            env["VLLM_TEST_EXTRA_ARGS"] = combined
        for key, value in self.mode.extra_env:
            env[key] = value
        return env

    def start(self) -> None:
        env = self.build_environment()
        command = build_server_command(env)
        self.command_repr = shlex.join(command)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            self.output_dir / "server_command.json",
            {
                "command": command,
                "environment": {
                    key: value
                    for key, value in env.items()
                    if key.startswith(
                        ("VLLM_TEST_", "VLLM_AGENT_", "VLLM_MCP_", "SUNCAKE_TRANSFER_")
                    )
                    or key
                    in {
                        "CUDA_VISIBLE_DEVICES",
                        "VLLM_USE_SIMPLE_KV_OFFLOAD",
                        "VLLM_LOG_STATS_INTERVAL",
                        "VLLM_TEMPORAL_SELECTION_POLICY",
                    }
                },
            },
        )
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            preexec_fn=os.setsid,
        )

    def stop(self) -> None:
        terminate_process_group(self.process)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


class MooncakeServerRunner(BaseServerRunner):
    def __init__(
        self,
        *,
        model: ModelSpec,
        output_dir: Path,
        port: int,
        cuda_devices: str,
        gpu_memory_utilization: float,
        swap_space: float,
        max_model_len: int,
        extra_server_args: str = "",
        mooncake_config_path: Optional[Path] = None,
        mooncake_master_port: Optional[int] = None,
        mooncake_global_segment_size: int = DEFAULT_MOONCAKE_GLOBAL_SEGMENT_SIZE,
        mooncake_local_buffer_size: int = DEFAULT_MOONCAKE_LOCAL_BUFFER_SIZE,
    ) -> None:
        self.model = model
        self.output_dir = output_dir
        self.port = port
        self.cuda_devices = cuda_devices
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space = swap_space
        self.max_model_len = max_model_len
        self.extra_server_args = extra_server_args
        self.mooncake_config_path = mooncake_config_path
        self.mooncake_master_port = mooncake_master_port
        self.mooncake_global_segment_size = mooncake_global_segment_size
        self.mooncake_local_buffer_size = mooncake_local_buffer_size
        self.process: Optional[subprocess.Popen[str]] = None
        self.log_handle = None
        self.log_path = output_dir / "mooncake_server.log"
        self.command_repr = ""

    def build_shell_command(self) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.mooncake_master_port is None:
            self.mooncake_master_port = choose_mooncake_master_port(self.port)
        master_server_address = f"127.0.0.1:{self.mooncake_master_port}"
        config_path = self.mooncake_config_path or write_mooncake_runtime_config(
            self.output_dir / "mooncake_runtime_config.json",
            global_segment_size=self.mooncake_global_segment_size,
            local_buffer_size=self.mooncake_local_buffer_size,
            master_server_address=master_server_address,
        )
        command = f"""
set -euo pipefail
source {shlex.quote(str(CONDA_SH))}
conda activate mooncake_agent
export CUDA_VISIBLE_DEVICES={shlex.quote(self.cuda_devices)}
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MOONCAKE_CONFIG_PATH={shlex.quote(str(config_path))}
export MOONCAKE_MASTER_PORT={self.mooncake_master_port}
echo "MOONCAKE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH"
echo "MOONCAKE_MASTER_PORT=$MOONCAKE_MASTER_PORT"
MASTER_PID=""
if ss -ltn | awk '{{print $4}}' | grep -Eq "(^|:|\\])${{MOONCAKE_MASTER_PORT}}$"; then
  echo "mooncake_master port $MOONCAKE_MASTER_PORT is already in use" >&2
  exit 1
fi
mooncake_master --port=$MOONCAKE_MASTER_PORT --logtostderr=1 > {shlex.quote(str(self.output_dir / "mooncake_master.log"))} 2>&1 &
MASTER_PID=$!
cleanup() {{
  if [[ -n "$MASTER_PID" ]]; then
    kill "$MASTER_PID" 2>/dev/null || true
    wait "$MASTER_PID" 2>/dev/null || true
  fi
}}
trap cleanup EXIT
python -m vllm.entrypoints.openai.api_server \
  --model {shlex.quote(self.model.path)} \
  --host 0.0.0.0 \
  --port {self.port} \
  --swap-space {self.swap_space} \
  --max-model-len {self.max_model_len} \
  --gpu-memory-utilization {self.gpu_memory_utilization} \
  --trust-remote-code \
  --kv-transfer-config '{{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}}' \
  {self.extra_server_args}
"""
        return command

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        shell_command = self.build_shell_command()
        self.command_repr = shell_command
        self.process = subprocess.Popen(
            ["bash", "-lc", shell_command],
            cwd=str(MOONCAKE_REPO),
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
            preexec_fn=os.setsid,
        )

    def stop(self) -> None:
        terminate_process_group(self.process)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


def make_server_runner(
    *,
    mode: ModeSpec,
    model: ModelSpec,
    output_dir: Path,
    port: int,
    cuda_devices: str,
    gpu_memory_utilization: float,
    swap_space: float,
    max_model_len: int,
    extra_server_args: str = "",
    num_gpu_blocks_override: Optional[int] = None,
) -> BaseServerRunner:
    if mode.runner_kind == "mooncake":
        return MooncakeServerRunner(
            model=model,
            output_dir=output_dir,
            port=port,
            cuda_devices=cuda_devices,
            gpu_memory_utilization=(
                mode.gpu_memory_utilization
                if mode.gpu_memory_utilization is not None
                else gpu_memory_utilization
            ),
            swap_space=swap_space,
            max_model_len=max_model_len,
            extra_server_args=extra_server_args,
        )
    return VllmServerRunner(
        mode=mode,
        model=model,
        output_dir=output_dir,
        port=port,
        cuda_devices=cuda_devices,
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=swap_space,
        max_model_len=max_model_len,
        extra_server_args=extra_server_args,
        num_gpu_blocks_override=num_gpu_blocks_override,
    )


def wait_for_server(port: int, timeout_s: int) -> None:
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + timeout_s
    with httpx.Client(timeout=30.0) as client:
        last_error = "server not ready"
        while time.time() < deadline:
            try:
                health = client.get(f"{base_url}/health")
                if health.status_code == 200:
                    return
                last_error = f"health={health.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(1.0)
    raise TimeoutError(
        f"Server on port {port} not ready within {timeout_s}s: {last_error}"
    )


def get_mcp_debug_state(port: int) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"http://127.0.0.1:{port}/v1/mcp/debug")
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _dict_int_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    keys = set(before) | set(after)
    delta: dict[str, int] = {}
    for key in keys:
        value = int(float(after.get(key, 0) or 0) - float(before.get(key, 0) or 0))
        if value:
            delta[key] = value
    return delta


def mcp_debug_metrics_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    if not before and not after:
        return {}
    event_counts = _dict_int_delta(
        before.get("coordination_event_counts", {}) or {},
        after.get("coordination_event_counts", {}) or {},
    )
    reason_counts = _dict_int_delta(
        before.get("coordination_reason_counts", {}) or {},
        after.get("coordination_reason_counts", {}) or {},
    )
    metric_totals = _dict_int_delta(
        before.get("coordination_metric_totals", {}) or {},
        after.get("coordination_metric_totals", {}) or {},
    )

    offload_decision_count = int(
        metric_totals.get(
            "offload_decision_count", event_counts.get("offload_decision", 0)
        )
    )
    offload_allowed_count = int(metric_totals.get("offload_allowed_count", 0))
    offload_rejected_count = int(
        metric_totals.get(
            "offload_rejected_count",
            max(0, offload_decision_count - offload_allowed_count),
        )
    )
    offload_reject_rate = (
        offload_rejected_count / offload_decision_count
        if offload_decision_count > 0
        else None
    )

    return {
        "coordinator_event_counts": event_counts,
        "coordinator_reason_counts": reason_counts,
        "offload_events": event_counts.get("offload_committed", 0),
        "upload_events": event_counts.get("upload_committed", 0),
        "offload_decision_count": offload_decision_count,
        "offload_allowed_count": offload_allowed_count,
        "offload_rejected_count": offload_rejected_count,
        "offload_reject_rate": offload_reject_rate,
        "offload_candidate_freed_blocks": int(
            metric_totals.get("offload_candidate_freed_blocks", 0)
        ),
        "offload_allowed_freed_blocks": int(
            metric_totals.get("offload_allowed_freed_blocks", 0)
        ),
        "offload_committed_blocks": int(
            metric_totals.get("offload_committed_blocks", 0)
        ),
        "upload_committed_blocks": int(metric_totals.get("upload_committed_blocks", 0)),
        "upload_reservation_count": int(
            metric_totals.get("upload_reservation_count", 0)
        ),
        "upload_reserved_blocks": int(metric_totals.get("upload_reserved_blocks", 0)),
    }


def run_client_case(
    *,
    model: ModelSpec,
    port: int,
    dataset: Path,
    task: str,
    request_rate: float,
    num_requests: int,
    seed: int,
    case_dir: Path,
    timeout_s: int,
    extra_client_args: tuple[str, ...] = (),
) -> tuple[float, Path, Path, Path]:
    case_dir.mkdir(parents=True, exist_ok=True)
    case_results_dir = case_dir / "app_results"
    case_results_dir.mkdir(parents=True, exist_ok=True)
    output_record_path = case_dir / "output_record.json"
    client_log_path = case_dir / "client.log"
    command_path = case_dir / "command.json"
    command = [
        sys.executable,
        "vllm_serving.py",
        "--port",
        str(port),
        "--model_path",
        model.path,
        "--dataset",
        str(dataset),
        "--task",
        task,
        "--request_rate",
        str(request_rate),
        "--num_requests",
        str(num_requests),
        "--seed",
        str(seed),
        "--output_dir",
        str(case_results_dir),
        "--output_file",
        str(output_record_path),
    ]
    command.extend(extra_client_args)
    extra_args_env = os.environ.get("REVISION_VLLM_SERVING_EXTRA_ARGS", "")
    if extra_args_env:
        command.extend(shlex.split(extra_args_env))
    write_json(command_path, {"command": command})

    start = time.time()
    with client_log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            cwd=str(ROOT),
            check=True,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    total_runtime_s = time.time() - start
    app_json_path = case_results_dir / f"app_qps_{request_rate}_num_{num_requests}.json"
    return total_runtime_s, app_json_path, client_log_path, command_path


def validate_case_logs(client_log_path: Path, server_slice: str) -> None:
    client_log_text = read_text(client_log_path)
    if "request error:" in client_log_text:
        raise RuntimeError(f"Client request errors detected in {client_log_path}")
    if " 400 Bad Request" in client_log_text or " 400 Bad Request" in server_slice:
        raise RuntimeError(
            f"HTTP 400 detected for case logs under {client_log_path.parent}"
        )
    if "maximum context length" in server_slice:
        raise RuntimeError(
            f"Context length validation failed for case logs under {client_log_path.parent}"
        )


def run_matrix(
    *,
    experiment: str,
    output_root: Path,
    modes: list[ModeSpec],
    tasks: list[str],
    qps_list: list[float],
    num_list: list[int],
    model: ModelSpec,
    dataset: Path,
    cuda_devices: str,
    gpu_memory_utilization: float,
    swap_space: float,
    max_model_len: int,
    port_base: int,
    server_ready_timeout: int,
    case_timeout: int,
    seed: int,
    extra_server_args: str = "",
    dry_run: bool = False,
) -> tuple[list[CaseSummary], list[BlockedSummary]]:
    """Run each case with a fresh server and retain its results and logs."""
    if not modes or not tasks or not qps_list or not num_list:
        raise ValueError(
            "Modes, tasks, QPS values, and request counts must be nonempty"
        )
    if set(tasks) - set(DEFAULT_TASKS):
        raise ValueError("Only code and research tasks are supported")
    if any(value <= 0 for value in qps_list + num_list):
        raise ValueError("QPS values and request counts must be positive")
    output_root = output_root.resolve()
    dataset = dataset.resolve()
    write_json(
        output_root / "plan.json",
        {
            "experiment": experiment,
            "model": asdict(model),
            "dataset": str(dataset),
            "modes": [asdict(mode) for mode in modes],
            "tasks": tasks,
            "qps_list": qps_list,
            "num_list": num_list,
            "seed": seed,
            "cuda_devices": cuda_devices,
            "gpu_memory_utilization": gpu_memory_utilization,
            "swap_space": swap_space,
            "max_model_len": max_model_len,
            "extra_server_args": extra_server_args,
            "python": sys.executable,
            "dry_run": dry_run,
        },
    )
    if dry_run:
        commands = {}
        for index, mode in enumerate(modes):
            if mode.runner_kind != "vllm":
                continue
            runner = make_server_runner(
                mode=mode,
                model=model,
                output_dir=output_root,
                port=port_base + index * 10,
                cuda_devices=cuda_devices,
                gpu_memory_utilization=gpu_memory_utilization,
                swap_space=swap_space,
                max_model_len=max_model_len,
                extra_server_args=extra_server_args,
            )
            commands[mode.name] = build_server_command(runner.build_environment())
        write_json(output_root / "server_commands.json", commands)
        return [], []
    all_cases: list[CaseSummary] = []
    blocked: list[BlockedSummary] = []

    for idx, mode in enumerate(modes):
        mode_dir = output_root / model.name / mode.name
        print(
            f"Running experiment '{experiment}' for model '{model.name}' in mode '{mode.name}'"
        )

        for task in tasks:
            for request_rate in qps_list:
                for num_requests in num_list:
                    case_name = f"{task}_qps_{request_rate:g}_num_{num_requests}"
                    case_dir = mode_dir / case_name
                    case_server_log = case_dir / "server.log"
                    port = choose_free_port(port_base + idx * 10)

                    print(
                        f"Running task '{task}' req num {num_requests} at {request_rate} qps on port {port}\n"
                    )

                    runner = make_server_runner(
                        mode=mode,
                        model=model,
                        output_dir=case_dir / "server_run",
                        port=port,
                        cuda_devices=cuda_devices,
                        gpu_memory_utilization=gpu_memory_utilization,
                        swap_space=swap_space,
                        max_model_len=max_model_len,
                        extra_server_args=extra_server_args,
                    )

                    def copy_case_server_log() -> str:
                        if not runner.log_path.exists():
                            return ""
                        server_log_text = runner.log_path.read_text(
                            encoding="utf-8", errors="ignore"
                        )
                        case_dir.mkdir(parents=True, exist_ok=True)
                        case_server_log.write_text(server_log_text, encoding="utf-8")
                        return server_log_text

                    try:
                        runner.start()
                        print("Start runner")
                        wait_for_server(port, server_ready_timeout)
                    except Exception as exc:
                        copy_case_server_log()
                        blocked_item = BlockedSummary(
                            experiment=experiment,
                            mode=mode.name,
                            reason="server_start_failed",
                            details={
                                "error": str(exc),
                                "exception_type": type(exc).__name__,
                                "model": model.name,
                                "task": task,
                                "request_rate": request_rate,
                                "num_requests": num_requests,
                                "case_dir": str(case_dir),
                                "server_log_path": str(case_server_log),
                                "command": getattr(runner, "command_repr", ""),
                                "traceback": traceback.format_exc(limit=8),
                            },
                        )
                        blocked.append(blocked_item)
                        write_json(case_dir / "blocked.json", asdict(blocked_item))
                        write_json(
                            output_root / "cases.json",
                            [asdict(case) for case in all_cases],
                        )
                        write_json(
                            output_root / "blocked.json",
                            [asdict(item) for item in blocked],
                        )
                        runner.stop()
                        print(
                            f"Server start failed and case will be skipped: "
                            f"mode={mode.name} task={task} "
                            f"qps={request_rate:g} num={num_requests}: {exc}"
                        )
                        print("=" * 100)
                        print("\n\n")
                        continue

                    collector = None
                    try:
                        if mode.runner_kind == "vllm":
                            collector = MetricsCollector(port, case_dir)
                            collector.start()
                        client_args = mode.extra_client_args + (
                            "--protocol",
                            "tokencake"
                            if mode.runner_kind == "vllm"
                            and (
                                mode.enable_agent_scheduling or mode.offloading_enabled
                            )
                            else "openai",
                        )
                        if not mode.offloading_enabled:
                            client_args += ("--disable_mcp_notifications",)
                        (
                            total_runtime_s,
                            app_json_path,
                            client_log_path,
                            command_path,
                        ) = run_client_case(
                            model=model,
                            port=port,
                            dataset=dataset,
                            task=task,
                            request_rate=request_rate,
                            num_requests=num_requests,
                            seed=seed,
                            case_dir=case_dir,
                            timeout_s=case_timeout,
                            extra_client_args=client_args,
                        )
                        runtime_metrics = (
                            collector.stop() if collector is not None else {}
                        )
                        collector = None
                        if runtime_metrics and not mode.enable_agent_scheduling:
                            runtime_metrics["scheduling"] = dict.fromkeys(
                                runtime_metrics["scheduling"]
                            )
                            runtime_metrics["critical_queue_wait_max_s"] = None
                            runtime_metrics["critical_growth_wait_max_s"] = None
                        server_slice = copy_case_server_log()
                        validate_case_logs(client_log_path, server_slice)
                        app_metrics = load_app_metrics(app_json_path, num_requests)
                        server_metrics = parse_server_slice(server_slice)
                        if runtime_metrics:
                            server_metrics.update(
                                legacy_summary_fields(runtime_metrics)
                            )
                        summary = CaseSummary(
                            experiment=experiment,
                            mode=mode.name,
                            task=task,
                            model_name=model.name,
                            model_path=model.path,
                            request_rate=request_rate,
                            num_requests=num_requests,
                            total_runtime_s=total_runtime_s,
                            throughput_rps=num_requests / total_runtime_s,
                            client_log_path=str(client_log_path),
                            server_log_path=str(case_server_log),
                            app_json_path=str(app_json_path),
                            command_path=str(command_path),
                            runtime_metrics=runtime_metrics,
                            metadata={
                                "command": getattr(runner, "command_repr", ""),
                                "runner_kind": mode.runner_kind,
                                "offloading_enabled": mode.offloading_enabled,
                                "scheduling_policy": mode.scheduling_policy,
                                "enable_agent_scheduling": mode.enable_agent_scheduling,
                                "mode_extra_env": dict(mode.extra_env),
                                "mode_extra_client_args": list(mode.extra_client_args),
                            },
                            **app_metrics,
                            **server_metrics,
                        )
                        write_json(case_dir / "summary.json", asdict(summary))
                        all_cases.append(summary)
                        write_json(
                            output_root / "cases.json",
                            [asdict(case) for case in all_cases],
                        )
                        write_json(
                            output_root / "blocked.json",
                            [asdict(item) for item in blocked],
                        )
                    except Exception as exc:
                        copy_case_server_log()
                        reason = "case_failed"
                        if isinstance(exc, subprocess.TimeoutExpired):
                            reason = "case_timeout"
                        elif isinstance(exc, subprocess.CalledProcessError):
                            reason = "client_failed"
                        blocked_item = BlockedSummary(
                            experiment=experiment,
                            mode=mode.name,
                            reason=reason,
                            details={
                                "error": str(exc),
                                "exception_type": type(exc).__name__,
                                "task": task,
                                "request_rate": request_rate,
                                "num_requests": num_requests,
                                "case_dir": str(case_dir),
                                "client_log_path": str(case_dir / "client.log"),
                                "server_log_path": str(case_server_log),
                                "command_path": str(case_dir / "command.json"),
                                "traceback": traceback.format_exc(limit=8),
                            },
                        )
                        blocked.append(blocked_item)
                        write_json(case_dir / "blocked.json", asdict(blocked_item))
                        write_json(
                            output_root / "cases.json",
                            [asdict(case) for case in all_cases],
                        )
                        write_json(
                            output_root / "blocked.json",
                            [asdict(item) for item in blocked],
                        )
                        print(
                            f"Case failed and will be skipped: mode={mode.name} "
                            f"task={task} qps={request_rate:g} "
                            f"num={num_requests} reason={reason}: {exc}"
                        )
                    finally:
                        if collector is not None:
                            try:
                                collector.stop()
                            except Exception as error:
                                write_json(
                                    case_dir / "metrics_error.json",
                                    {"error": str(error)},
                                )
                        runner.stop()
                        print("=" * 100)
                        print("\n\n")

    write_json(output_root / "cases.json", [asdict(case) for case in all_cases])
    write_json(output_root / "blocked.json", [asdict(item) for item in blocked])
    return all_cases, blocked


def write_markdown_cases(
    path: Path, title: str, cases: list[CaseSummary], blocked: list[BlockedSummary]
) -> None:
    lines = [f"# {title}", ""]
    if blocked:
        lines.extend(["## Blocked", ""])
        for item in blocked:
            lines.append(
                f"- `{item.mode}`: {item.reason} - {item.details.get('error', '')}"
            )
        lines.append("")
    lines.extend(["## Cases", ""])
    for case in cases:
        reject_rate = (
            ""
            if case.offload_reject_rate is None
            else f" reject=`{case.offload_reject_rate:.2%}`"
        )
        lines.append(
            f"- `{case.mode}` `{case.task}` qps=`{case.request_rate}` num=`{case.num_requests}`: "
            f"avg=`{case.avg_app_latency_s:.3f}s` p95=`{case.p95_app_latency_s:.3f}s` "
            f"throughput=`{case.throughput_rps:.3f} rps` offload=`{case.offload_events}` "
            f"upload=`{case.upload_events}` candidates=`{case.offload_decision_count}`"
            f"{reject_rate} swap=`{case.swap_in_count}/{case.swap_out_count}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
