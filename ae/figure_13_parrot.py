#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.app.code_writer_mcp import (
    CodeWriterMcpApplication, CodeContextBuilder)
from agent.app.deep_research_mcp import (
    DeepResearchMcpApplication, ResearchPressureContextBuilder)
from agent.app.request import ApplicationRequest
from agent.graph.meta import LLMCallMetadata
from agent.vllm_prompt import CODE_WRITER_PROMPT
from ae.common import CONDA_SH, PARROT_REPO, get_model_specs, now_stamp, write_json
from ae.parrot_graph_client import ParrotGraphExecutor


_GRAPH_BUILD_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Figure 13: run application graphs on Parrot semantic APIs.")
    parser.add_argument("--output-root",
                        type=Path,
                        default=Path("ae/results/figure_13_parrot"))
    parser.add_argument("--conda-env", default="parrot_cake")
    parser.add_argument("--engine-config",
                        default="sample_configs/engine/qwen2.5-7b-instruct-local.json")
    parser.add_argument("--core-config",
                        default="sample_configs/core/localhost_serve_core.json")
    parser.add_argument("--model-path",
                        default=get_model_specs()["qwen2.5-7b-instruct"].path)
    parser.add_argument("--task",
                        default="code",
                        help=("Task list: code, research, both, "
                              "or comma-separated values."))
    parser.add_argument("--dataset",
                        type=Path,
                        default=Path("dataset/agentcodeclean_new.json"))
    parser.add_argument("--num-requests",
                        default="20",
                        help="Request count or comma-separated list, e.g. 20,30.")
    parser.add_argument("--request-rate",
                        default="1.0",
                        help="QPS or comma-separated list, e.g. 0.05,0.1,1.0.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument("--core-port",
                        type=int,
                        default=0,
                        help="Parrot ServeCore port. Use 0 for auto allocation.")
    parser.add_argument("--engine-port",
                        type=int,
                        default=0,
                        help="Parrot engine port. Use 0 for auto allocation.")
    parser.add_argument("--port-base",
                        type=int,
                        default=9000,
                        help="First port to scan when auto-allocating Parrot ports.")
    parser.add_argument("--port-search-size",
                        type=int,
                        default=2000,
                        help="Number of ports to scan for auto allocation.")
    parser.add_argument("--core-ready-timeout", type=int, default=60)
    parser.add_argument("--engine-ready-timeout", type=int, default=240)
    parser.add_argument("--probe-timeout", type=int, default=180)
    parser.add_argument("--case-timeout", type=int, default=7200)
    parser.add_argument("--max-new-tokens-cap", type=int, default=0)
    parser.add_argument("--parrot-max-total-tokens", type=int, default=32768)
    parser.add_argument("--parrot-max-num-batched-tokens",
                        type=int,
                        default=32768)
    parser.add_argument("--parrot-num-kv-cache-blocks", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization",
                        "--parrot-gpu-memory-utilization",
                        dest="gpu_memory_utilization",
                        type=float,
                        default=0.0,
                        help=("Fraction of GPU memory Parrot may use. "
                              "When >0, Parrot auto-computes KV cache blocks."))
    parser.add_argument("--node-concurrency", type=int, default=1)
    parser.add_argument("--llm-concurrency", type=int, default=1)
    parser.add_argument("--tool-noise-profile",
                        choices=["none", "gaussian", "uniform"],
                        default="none")
    parser.add_argument("--tool-noise-scale", type=float, default=0.0)
    parser.add_argument("--output-criteria", default="latency")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Build app graphs and print stats without launching Parrot.")
    parser.add_argument("--reuse-server",
                        action="store_true",
                        help="Use an already running Parrot ServeCore instead of launching.")
    parser.add_argument("--parrot-url", default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv_ints(value: str) -> list[int]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer value")
    values = [int(item) for item in items]
    if any(item <= 0 for item in values):
        raise ValueError(f"Integer list must be positive: {value}")
    return values


def parse_csv_floats(value: str) -> list[float]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one float value")
    values = [float(item) if item != "inf" else float("inf") for item in items]
    if any(item <= 0 for item in values):
        raise ValueError(f"Float list must be positive: {value}")
    return values


def selected_tasks(task_arg: str) -> list[str]:
    values: list[str] = []
    for item in str(task_arg).split(","):
        task = item.strip()
        if not task:
            continue
        if task == "both":
            values.extend(["code", "research"])
        elif task in {"code", "research"}:
            values.append(task)
        else:
            raise ValueError(
                f"Unsupported task {task!r}; expected code, research, or both.")
    if not values:
        raise ValueError("Expected at least one task")
    return list(dict.fromkeys(values))


@dataclass(frozen=True)
class CaseSpec:
    task: str
    num_requests: int
    request_rate: float

    @property
    def name(self) -> str:
        return f"{self.task}_qps_{self.request_rate:g}_num_{self.num_requests}"


def build_case_matrix(args: argparse.Namespace) -> list[CaseSpec]:
    tasks = selected_tasks(args.task)
    nums = parse_csv_ints(args.num_requests)
    rates = parse_csv_floats(args.request_rate)
    return [
        CaseSpec(task=task, num_requests=num_requests, request_rate=request_rate)
        for task in tasks
        for num_requests in nums
        for request_rate in rates
    ]


def create_run_root(output_root: Path) -> Path:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = now_stamp()
    candidates = [output_root / stamp]
    candidates.extend(output_root / f"{stamp}_{os.getpid()}_{i}" for i in range(100))
    for candidate in candidates:
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create unique run directory under {output_root}")


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._handle.write(f"[{timestamp}] {message}\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def write_server_log(log_dir: Path) -> None:
    server_log = log_dir / "server.log"
    sources = [
        log_dir / "core.launch.log",
        log_dir / "core.log",
        log_dir / "core_stdout.out",
        log_dir / "engine.launch.log",
        log_dir / "engine.log",
        log_dir / "engine_stdout.out",
    ]
    with server_log.open("w", encoding="utf-8") as out:
        wrote_any = False
        for source in sources:
            if not source.exists():
                continue
            wrote_any = True
            out.write(f"\n===== {source.name} =====\n")
            with source.open("r", encoding="utf-8", errors="ignore") as src:
                for line in src:
                    out.write(line)
        if not wrote_any:
            out.write("No managed Parrot server logs were captured.\n")


@contextlib.contextmanager
def capture_graph_build_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _GRAPH_BUILD_LOCK:
        with log_path.open("a", encoding="utf-8") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(
                    handle):
                yield


def _port_registry_paths() -> tuple[Path, Path]:
    state_dir = Path(
        os.environ.get("TOKENCAKE_PARROT_PORT_STATE_DIR", "/tmp")).resolve()
    suffix = os.environ.get("USER") or str(os.getuid())
    return (
        state_dir / f"tokencake_parrot_ports_{suffix}.json",
        state_dir / f"tokencake_parrot_ports_{suffix}.lock",
    )


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_port_bindable(port: int) -> bool:
    if port <= 0 or port > 65535:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _load_port_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ports": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ports": {}}
    if not isinstance(data, dict) or not isinstance(data.get("ports"), dict):
        return {"ports": {}}
    return data


def _write_port_registry(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True),
                        encoding="utf-8")
    tmp_path.replace(path)


def _clean_port_registry(data: dict[str, Any]) -> set[int]:
    ports = data.setdefault("ports", {})
    stale: list[str] = []
    for port_str, entry in ports.items():
        try:
            pid = int(entry.get("pid", -1))
        except (AttributeError, TypeError, ValueError):
            stale.append(port_str)
            continue
        if not _is_pid_alive(pid):
            stale.append(port_str)
    for port_str in stale:
        ports.pop(port_str, None)
    return {int(port) for port in ports}


class PortReservation:
    def __init__(self, *, core_port: int, engine_port: int,
                 registry_path: Path) -> None:
        self.core_port = core_port
        self.engine_port = engine_port
        self.registry_path = registry_path
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        registry_path = self.registry_path
        _, lock_path = _port_registry_paths()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            data = _load_port_registry(registry_path)
            ports = data.setdefault("ports", {})
            pid = os.getpid()
            for port in (self.core_port, self.engine_port):
                entry = ports.get(str(port))
                if isinstance(entry, dict) and entry.get("pid") == pid:
                    ports.pop(str(port), None)
            _write_port_registry(registry_path, data)
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        self._released = True


def reserve_parrot_ports(*, requested_core_port: int,
                         requested_engine_port: int, port_base: int,
                         search_size: int,
                         run_root: Path) -> PortReservation:
    if requested_core_port < 0 or requested_engine_port < 0:
        raise ValueError("Parrot ports must be >= 0. Use 0 for auto allocation.")
    if port_base <= 0 or port_base > 65535:
        raise ValueError(f"Invalid --port-base: {port_base}")
    if search_size <= 0:
        raise ValueError(f"Invalid --port-search-size: {search_size}")

    registry_path, lock_path = _port_registry_paths()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        data = _load_port_registry(registry_path)
        reserved_ports = _clean_port_registry(data)

        def available(port: int) -> bool:
            return port not in reserved_ports and _is_port_bindable(port)

        def candidate_pairs() -> list[tuple[int, int]]:
            max_port = min(65535, port_base + search_size - 1)
            if requested_core_port > 0 and requested_engine_port > 0:
                return [(requested_core_port, requested_engine_port)]
            if requested_core_port > 0:
                start = max(port_base, requested_core_port + 1)
                return [(requested_core_port, port)
                        for port in range(start, max_port + 1)
                        if port != requested_core_port]
            if requested_engine_port > 0:
                return [(port, requested_engine_port)
                        for port in range(port_base, max_port + 1)
                        if port != requested_engine_port]
            return [(port, port + 1)
                    for port in range(port_base, max_port, 2)]

        selected: Optional[tuple[int, int]] = None
        for core_port, engine_port in candidate_pairs():
            if core_port == engine_port:
                continue
            if available(core_port) and available(engine_port):
                selected = (core_port, engine_port)
                break

        if selected is None:
            raise RuntimeError(
                "Could not reserve Parrot core/engine ports "
                f"(core={requested_core_port}, engine={requested_engine_port}, "
                f"base={port_base}, search_size={search_size}).")

        now = time.time()
        pid = os.getpid()
        core_port, engine_port = selected
        ports = data.setdefault("ports", {})
        ports[str(core_port)] = {
            "pid": pid,
            "role": "core",
            "run_root": str(run_root),
            "allocated_at": now,
        }
        ports[str(engine_port)] = {
            "pid": pid,
            "role": "engine",
            "run_root": str(run_root),
            "allocated_at": now,
        }
        _write_port_registry(registry_path, data)
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    return PortReservation(core_port=core_port,
                           engine_port=engine_port,
                           registry_path=registry_path)


class ManagedProcess:
    def __init__(self,
                 *,
                 name: str,
                 command: str,
                 cwd: Path,
                 log_path: Path,
                 env: dict[str, str]) -> None:
        self.name = name
        self.command = command
        self.cwd = cwd
        self.log_path = log_path
        self.env = env
        self.process: Optional[subprocess.Popen[str]] = None
        self._log_handle = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            ["bash", "-lc", self.command],
            cwd=str(self.cwd),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=self.env,
            preexec_fn=os.setsid,
        )

    def ensure_running(self) -> None:
        if self.process is None:
            raise RuntimeError(f"{self.name} has not been started")
        return_code = self.process.poll()
        if return_code is not None:
            raise RuntimeError(f"{self.name} exited early with code {return_code}")

    def stop(self) -> None:
        if self.process is not None:
            try:
                pgid = os.getpgid(self.process.pid)
            except ProcessLookupError:
                pgid = None
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=10)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


class ParrotLauncher:
    def __init__(self,
                 *,
                 run_dir: Path,
                 conda_env: str,
                 core_config: Path,
                 engine_config: Path,
                 model_path: str,
                 cuda_devices: str,
                 core_port: int,
                 engine_port: int,
                 max_total_tokens: int,
                 max_num_batched_tokens: int,
                 num_kv_cache_blocks: int,
                 gpu_memory_utilization: float) -> None:
        self.run_dir = run_dir
        self.conda_env = conda_env
        self.core_config_src = core_config
        self.engine_config_src = engine_config
        self.model_path = model_path
        self.cuda_devices = cuda_devices
        self.core_port = core_port
        self.engine_port = engine_port
        self.max_total_tokens = max_total_tokens
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_kv_cache_blocks = num_kv_cache_blocks
        self.gpu_memory_utilization = gpu_memory_utilization
        self.core_log = run_dir / "core.log"
        self.engine_log = run_dir / "engine.log"
        self.core_launch_log = run_dir / "core.launch.log"
        self.engine_launch_log = run_dir / "engine.launch.log"
        self.core_config_path = run_dir / "parrot_core_config.json"
        self.engine_config_path = run_dir / "parrot_engine_config.json"
        self.core_proc: Optional[ManagedProcess] = None
        self.engine_proc: Optional[ManagedProcess] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.core_port}"

    def write_runtime_configs(self) -> None:
        core_config = read_json(self.core_config_src)
        engine_config = read_json(self.engine_config_src)
        core_config["host"] = "localhost"
        core_config["port"] = self.core_port
        engine_config["host"] = "localhost"
        engine_config["port"] = self.engine_port
        engine_config["model"] = self.model_path
        engine_config["tokenizer"] = self.model_path
        engine_config.setdefault("serve_core", {})
        engine_config["serve_core"]["host"] = "localhost"
        engine_config["serve_core"]["port"] = self.core_port
        if self.max_total_tokens > 0:
            engine_config.setdefault("scheduler", {})
            engine_config["scheduler"]["max_total_tokens"] = self.max_total_tokens
        if self.max_num_batched_tokens > 0:
            engine_config.setdefault("scheduler", {})
            engine_config["scheduler"]["max_num_batched_tokens"] = (
                self.max_num_batched_tokens)
        if self.gpu_memory_utilization > 0:
            engine_config.setdefault("instance", {})
            engine_config["instance"]["gpu_memory_utilization"] = (
                self.gpu_memory_utilization)
            engine_config["instance"].pop("num_kv_cache_blocks", None)
        elif self.num_kv_cache_blocks > 0:
            engine_config.setdefault("instance", {})
            engine_config["instance"]["num_kv_cache_blocks"] = (
                self.num_kv_cache_blocks)
        write_json(self.core_config_path, core_config)
        write_json(self.engine_config_path, engine_config)

    def start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.write_runtime_configs()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_devices
        env["SIMULATE_NETWORK_LATENCY_PRT"] = "0"

        core_cmd = (
            f"source {shlex.quote(str(CONDA_SH))} && "
            f"conda activate {shlex.quote(self.conda_env)} && "
            f"python3 -m parrot.serve.http_server "
            f"--config_path {shlex.quote(str(self.core_config_path))} "
            f"--log_dir {shlex.quote(str(self.run_dir))} "
            f"--log_filename {shlex.quote(self.core_log.name)}")
        engine_cmd = (
            f"source {shlex.quote(str(CONDA_SH))} && "
            f"conda activate {shlex.quote(self.conda_env)} && "
            f"python3 -m parrot.engine.http_server "
            f"--config_path {shlex.quote(str(self.engine_config_path))} "
            f"--log_dir {shlex.quote(str(self.run_dir))} "
            f"--log_filename {shlex.quote(self.engine_log.name)}")
        self.core_proc = ManagedProcess(name="parrot core",
                                        command=core_cmd,
                                        cwd=PARROT_REPO,
                                        log_path=self.core_launch_log,
                                        env=env)
        self.engine_proc = ManagedProcess(name="parrot engine",
                                          command=engine_cmd,
                                          cwd=PARROT_REPO,
                                          log_path=self.engine_launch_log,
                                          env=env)
        self.core_proc.start()
        self.engine_proc.start()

    def stop(self) -> None:
        if self.engine_proc is not None:
            self.engine_proc.stop()
        if self.core_proc is not None:
            self.core_proc.stop()

    def _ensure_running(self) -> None:
        if self.core_proc is not None:
            self.core_proc.ensure_running()
        if self.engine_proc is not None:
            self.engine_proc.ensure_running()

    def wait_for_tcp(self, timeout_s: int) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self._ensure_running()
            try:
                with socket.create_connection(("127.0.0.1", self.core_port),
                                              timeout=1.0):
                    return
            except OSError:
                time.sleep(1.0)
        raise TimeoutError(f"Parrot core tcp port {self.core_port} not ready")

    def wait_for_engine(self, timeout_s: int) -> None:
        engine_config = read_json(self.engine_config_path)
        needle = f"Engine {engine_config['engine_name']} (id="
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self._ensure_running()
            if self.core_log.exists():
                text = self.core_log.read_text(encoding="utf-8",
                                               errors="ignore")
                if needle in text:
                    return
            time.sleep(1.0)
        raise TimeoutError(f"Engine registration log not observed: {needle}")

    def probe(self, timeout_s: int) -> None:
        deadline = time.time() + timeout_s
        payload = {
            "model": self.model_path,
            "prompt": "Hello",
            "max_tokens": 1,
            "temperature": 0.0,
        }
        last_error = "not attempted"
        with httpx.Client(timeout=60.0) as client:
            while time.time() < deadline:
                self._ensure_running()
                try:
                    response = client.post(f"{self.url}/v1/common_inference",
                                           json=payload)
                    response.raise_for_status()
                    data = response.json()
                    if isinstance(data, dict) and "choices" in data:
                        return
                    last_error = f"unexpected response: {data}"
                except Exception as exc:
                    last_error = str(exc)
                time.sleep(1.0)
        raise TimeoutError(f"Parrot probe failed: {last_error}")


def load_dataset(path: Path, num_requests: int) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        prompts = [data[str(i % len(data))]["prompt"] for i in range(num_requests)]
    else:
        prompts = [data[i % len(data)]["prompt"] for i in range(num_requests)]
    return prompts


@lru_cache(maxsize=4)
def pressure_context_builder(task: str, model_path: str):
    from transformers import AutoTokenizer

    builders = {"code": CodeContextBuilder,
                "research": ResearchPressureContextBuilder}
    if task not in builders:
        raise ValueError(f"Unsupported task: {task}")
    return builders[task](AutoTokenizer.from_pretrained(model_path), ROOT)


def build_application(task: str, prompt: str, model_path: str) -> ApplicationRequest:
    llm_metadata = LLMCallMetadata(model="gpt-4o-mini",
                                   max_new_tokens=500,
                                   temperature=0)
    context = pressure_context_builder(task, model_path).build(prompt)
    if task == "code":
        application = CodeWriterMcpApplication
    elif task == "research":
        application = DeepResearchMcpApplication
    else:
        raise ValueError(f"Unsupported task: {task}")
    app = application(llm_metadata, prompt=context.text,
                      context_token_count=context.token_count,
                      context_sources=context.sources)
    return ApplicationRequest(prompt=CODE_WRITER_PROMPT, application=app)


def graph_stats(task: str, prompt: str, model_path: str) -> dict[str, Any]:
    app_req = build_application(task, prompt, model_path)
    graph = app_req.application.graph
    by_type: dict[str, int] = {}
    mcp_count = 0
    for node in graph.nodes.values():
        key = str(getattr(node, "node_type", None))
        by_type[key] = by_type.get(key, 0) + 1
        if hasattr(node, "mcp_function"):
            mcp_count += 1
    return {
        "task": task,
        "nodes": len(graph.nodes),
        "edges": sum(len(items) for items in graph.edges.values()),
        "mcp_nodes": mcp_count,
        "node_types": by_type,
    }


def build_graph_stats(task_list: list[str], prompt: str,
                      log_path: Path, model_path: str) -> list[dict[str, Any]]:
    with capture_graph_build_output(log_path):
        return [graph_stats(task, prompt, model_path) for task in task_list]


async def run_one_case(args: argparse.Namespace, *, case: CaseSpec,
                       parrot_url: str, output_dir: Path,
                       logger: Optional[RunLogger]) -> dict[str, Any]:
    prompts = load_dataset(args.dataset, case.num_requests)
    app_results_dir = output_dir / "app_results"
    app_results_dir.mkdir(parents=True, exist_ok=True)
    executor = ParrotGraphExecutor(
        parrot_url=parrot_url,
        model_path=args.model_path,
        max_new_tokens_cap=args.max_new_tokens_cap,
        tool_noise_profile=args.tool_noise_profile,
        tool_noise_scale=args.tool_noise_scale,
        output_criteria=args.output_criteria,
        node_concurrency=args.node_concurrency,
        llm_concurrency=args.llm_concurrency,
    )
    results: dict[str, dict[str, Any]] = {}
    app_latencies: list[float] = []
    all_start = time.time()
    if logger is not None:
        logger.log(
            f"case={case.name} start task={case.task} num_requests={case.num_requests} qps={case.request_rate:g}"
        )

    async def schedule_request(index: int, prompt: str,
                               expected_start: float) -> None:
        delay = expected_start - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        with capture_graph_build_output(output_dir / "graph_build.log"):
            app_req = build_application(case.task, prompt, args.model_path)
        result = await executor.run_application(app_req,
                                                all_start_time=all_start)
        results[str(index)] = asdict(result)
        app_latencies.append(result.app_latency)
        progress = (
            f"[{case.task} qps={case.request_rate:g} num={case.num_requests}] "
            f"finished {len(app_latencies)}/{case.num_requests}"
        )
        if logger is not None:
            logger.log(progress)

    interval = 0.0 if case.request_rate == float("inf") else 1.0 / case.request_rate
    tasks = [
        asyncio.create_task(
            schedule_request(i, prompt, all_start + i * interval))
        for i, prompt in enumerate(prompts)
    ]
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=args.case_timeout)

    app_json = app_results_dir / f"{case.name}.json"
    write_json(app_json, results)
    total_runtime = time.time() - all_start
    summary = {
        "experiment": "figure_13",
        "case": case.name,
        "task": case.task,
        "mode": "parrot_graph",
        "num_requests": case.num_requests,
        "request_rate": case.request_rate,
        "total_runtime_s": total_runtime,
        "throughput_rps": case.num_requests / total_runtime,
        "avg_app_latency_s": float(np.mean(app_latencies)),
        "p50_app_latency_s": float(np.percentile(app_latencies, 50)),
        "p90_app_latency_s": float(np.percentile(app_latencies, 90)),
        "p95_app_latency_s": float(np.percentile(app_latencies, 95)),
        "p99_app_latency_s": float(np.percentile(app_latencies, 99)),
        "max_app_latency_s": float(np.max(app_latencies)),
        "app_json_path": str(app_json),
    }
    if logger is not None:
        logger.log(
            f"case={case.name} complete runtime_s={total_runtime:.3f} output={app_json}"
        )
    return summary


def start_case_launcher(args: argparse.Namespace, *, case: CaseSpec,
                        case_dir: Path,
                        logger: RunLogger) -> tuple[ParrotLauncher,
                                                    PortReservation, str]:
    port_reservation = reserve_parrot_ports(
        requested_core_port=args.core_port,
        requested_engine_port=args.engine_port,
        port_base=args.port_base,
        search_size=args.port_search_size,
        run_root=case_dir,
    )
    launcher: Optional[ParrotLauncher] = None
    try:
        core_port = port_reservation.core_port
        engine_port = port_reservation.engine_port
        logger.log(
            f"case={case.name} reserved Parrot ports core_port={core_port} engine_port={engine_port}"
        )
        launcher = ParrotLauncher(
            run_dir=case_dir,
            conda_env=args.conda_env,
            core_config=PARROT_REPO / args.core_config,
            engine_config=PARROT_REPO / args.engine_config,
            model_path=args.model_path,
            cuda_devices=args.cuda_devices,
            core_port=core_port,
            engine_port=engine_port,
            max_total_tokens=args.parrot_max_total_tokens,
            max_num_batched_tokens=args.parrot_max_num_batched_tokens,
            num_kv_cache_blocks=args.parrot_num_kv_cache_blocks,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        logger.log(
            f"case={case.name} starting Parrot core_port={core_port} engine_port={engine_port} cuda={args.cuda_devices}"
        )
        launcher.start()
        launcher.wait_for_tcp(args.core_ready_timeout)
        logger.log(f"case={case.name} Parrot core TCP ready")
        launcher.wait_for_engine(args.engine_ready_timeout)
        logger.log(f"case={case.name} Parrot engine registered")
        launcher.probe(args.probe_timeout)
        logger.log(f"case={case.name} Parrot common_inference probe succeeded")
        return launcher, port_reservation, launcher.url
    except Exception:
        if launcher is not None:
            launcher.stop()
        port_reservation.release()
        write_server_log(case_dir)
        raise


def run_reuse_server_case(args: argparse.Namespace, *, case: CaseSpec,
                          case_dir: Path,
                          logger: RunLogger) -> dict[str, Any]:
    if not args.parrot_url:
        raise ValueError("--reuse-server requires --parrot-url")
    logger.log(f"case={case.name} reusing Parrot at {args.parrot_url}")
    summary = asyncio.run(
        run_one_case(args,
                     case=case,
                     parrot_url=args.parrot_url,
                     output_dir=case_dir,
                     logger=logger))
    write_json(case_dir / "server.log", {
        "status": "reuse_server",
        "parrot_url": args.parrot_url,
    })
    return summary


def write_progress(run_root: Path, all_summaries: list[dict[str, Any]],
                   blocked: list[dict[str, Any]]) -> None:
    write_json(run_root / "cases.json", all_summaries)
    write_json(run_root / "blocked.json", blocked)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    run_root = create_run_root(args.output_root)
    logger = RunLogger(run_root / "client.log")
    all_summaries: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    try:
        logger.log(f"run_dir={run_root}")
        logger.log(
            f"args={json.dumps(vars(args), default=str, sort_keys=True)}")
        case_matrix = build_case_matrix(args)
        write_json(run_root / "plan.json", {
            "experiment": "figure_13",
            "dry_run": args.dry_run,
            "cases": [asdict(case) for case in case_matrix],
        })
        task_list = selected_tasks(args.task)
        logger.log(
            f"case_matrix={json.dumps([asdict(case) for case in case_matrix], sort_keys=True)}"
        )
        first_prompt = load_dataset(args.dataset, 1)[0]
        stats = build_graph_stats(task_list, first_prompt,
                                  run_root / "graph_build.log", args.model_path)
        write_json(run_root / "graph_stats.json", stats)
        logger.log(f"graph_stats={json.dumps(stats, sort_keys=True)}")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "cases": [asdict(case) for case in case_matrix],
                        "graph_stats": stats,
                    },
                    indent=2,
                    sort_keys=True,
                ))
            print(run_root)
            logger.log("dry_run complete")
            return 0

        for case in case_matrix:
            case_dir = run_root / case.task / case.name
            case_dir.mkdir(parents=True, exist_ok=True)
            case_logger = RunLogger(case_dir / "client.log")
            launcher: Optional[ParrotLauncher] = None
            port_reservation: Optional[PortReservation] = None
            parrot_url: Optional[str] = None
            try:
                case_logger.log(f"run_dir={run_root}")
                case_logger.log(f"case_dir={case_dir}")
                case_logger.log(
                    f"args={json.dumps(vars(args), default=str, sort_keys=True)}"
                )
                case_logger.log(
                    f"case={json.dumps(asdict(case), sort_keys=True)}")
                logger.log(f"case={case.name} start")

                if args.reuse_server:
                    summary = run_reuse_server_case(args,
                                                    case=case,
                                                    case_dir=case_dir,
                                                    logger=case_logger)
                    parrot_url = args.parrot_url
                else:
                    launcher, port_reservation, parrot_url = start_case_launcher(
                        args, case=case, case_dir=case_dir, logger=case_logger)
                    summary = asyncio.run(
                        run_one_case(args,
                                     case=case,
                                     parrot_url=parrot_url,
                                     output_dir=case_dir,
                                     logger=case_logger))

                summary.setdefault("client_log_path", str(case_dir / "client.log"))
                summary.setdefault("server_log_path", str(case_dir / "server.log"))
                write_json(case_dir / "summary.json", summary)
                all_summaries.append(summary)
                write_progress(run_root, all_summaries, blocked)
                logger.log(f"case={case.name} success")
            except Exception as exc:
                blocked_item = {
                    "status": "blocked",
                    "reason": "case_failed",
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                    "case": asdict(case),
                    "case_dir": str(case_dir),
                    "parrot_url": parrot_url,
                    "client_log_path": str(case_dir / "client.log"),
                    "server_log_path": str(case_dir / "server.log"),
                    "traceback": traceback.format_exc(limit=8),
                }
                blocked.append(blocked_item)
                write_json(case_dir / "blocked.json", blocked_item)
                write_progress(run_root, all_summaries, blocked)
                case_logger.log(
                    f"case={case.name} blocked error={exc}")
                logger.log(f"case={case.name} blocked error={exc}")
            finally:
                if launcher is not None:
                    launcher.stop()
                if port_reservation is not None:
                    port_reservation.release()
                if not args.reuse_server:
                    write_server_log(case_dir)
                case_logger.close()

        write_json(run_root / "summary.json", {
            "status": "success",
            "blocked_count": len(blocked),
            "tasks": task_list,
            "case_matrix": [asdict(case) for case in case_matrix],
            "cases": all_summaries,
            "blocked": blocked,
            "graph_stats": stats,
        })
        logger.log(
            f"run status=success cases={len(all_summaries)} blocked={len(blocked)}"
        )
        print(run_root)
        return 0 if not blocked else 1
    except Exception as exc:
        blocked_tasks = locals().get("task_list", [])
        blocked_cases = locals().get("case_matrix", [])
        write_json(run_root / "blocked.json", {
            "status": "blocked",
            "error": str(exc),
            "parrot_url": locals().get("parrot_url"),
            "tasks": blocked_tasks,
            "case_matrix": [asdict(case) for case in blocked_cases],
        })
        logger.log(f"run status=blocked error={exc}")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
