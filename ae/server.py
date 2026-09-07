"""Translate the retained AE launch settings into current vLLM arguments."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections.abc import Mapping


def additional_config(env: Mapping[str, str]) -> dict:
    scheduling = env.get("VLLM_TEST_ENABLE_AGENT_SCHEDULING", "0") == "1"
    offload = env.get("VLLM_TEST_ENABLE_KVCACHE_CPU_OFFLOADING", "0") == "1"
    if not (scheduling or offload):
        return {}
    schedule_settings = {"enabled": scheduling}
    offload_settings = {"enabled": offload}
    if scheduling:
        fields = {
            "VLLM_TEMPORAL_SELECTION_POLICY": ("temporal_selection", str),
            "VLLM_AGENT_IMPORTANT_AGENT_RATIO": ("critical_ratio", float),
            "VLLM_AGENT_GPU_USAGE_HIGH_WATERMARK": ("gpu_usage_high", float),
            "VLLM_AGENT_GPU_USAGE_LOW_WATERMARK": ("gpu_usage_low", float),
            "VLLM_AGENT_RESERVE_ADJUSTMENT_STEP": ("reserve_adjustment_step", float),
        }
        for key, (field, convert) in fields.items():
            if env.get(key):
                schedule_settings[field] = convert(env[key])
    if offload:
        transfer = {}
        if env.get("SUNCAKE_TRANSFER_BANDWIDTH_GBPS"):
            bandwidth = float(env["SUNCAKE_TRANSFER_BANDWIDTH_GBPS"])
            if bandwidth <= 0:
                raise ValueError("Transfer bandwidth must be positive (decimal GB/s)")
            transfer.update(d2h_bandwidth_gbps=bandwidth, h2d_bandwidth_gbps=bandwidth)
        if env.get("VLLM_MCP_TRANSFER_BASE_TIME_S"):
            base = float(env["VLLM_MCP_TRANSFER_BASE_TIME_S"])
            transfer.update(d2h_base_time_s=base, h2d_base_time_s=base)
        if transfer:
            offload_settings["transfer"] = transfer
    return {"tokencake": {"scheduling": schedule_settings, "offload": offload_settings}}


def build_command(env: Mapping[str, str]) -> list[str]:
    model = env.get("VLLM_TEST_MODEL_PATH")
    if not model:
        raise ValueError("Set VLLM_TEST_MODEL_PATH to a model directory or model ID")
    if env.get("VLLM_MCP_TRANSFER_TIME_PER_BLOCK_S"):
        raise ValueError(
            "Use SUNCAKE_TRANSFER_BANDWIDTH_GBPS; transfer costs now use bytes"
        )
    if env.get("VLLM_MCP_LOG_ALL_COORDINATION_EVENTS") == "1":
        raise ValueError("MCP decision logs were replaced by the /metrics counters")
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--host",
        env.get("VLLM_TEST_HOST", "127.0.0.1"),
        "--port",
        env.get("VLLM_TEST_PORT", "8055"),
        "--scheduling-policy",
        "fcfs",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--uvicorn-log-level",
        env.get("VLLM_TEST_UVICORN_LOG_LEVEL", "info"),
    ]
    fields = {
        "VLLM_TEST_GPU_MEMORY_UTILIZATION": "--gpu-memory-utilization",
        "VLLM_TEST_MAX_MODEL_LEN": "--max-model-len",
        "VLLM_TEST_NUM_GPU_BLOCKS_OVERRIDE": "--num-gpu-blocks-override",
        "VLLM_TEST_SCHEDULER_CLS": "--scheduler-cls",
    }
    for key, flag in fields.items():
        if env.get(key):
            command.extend([flag, env[key]])
    config = additional_config(env)
    if config:
        command.extend(["--additional-config", json.dumps(config, sort_keys=True)])
    if env.get("VLLM_TEST_ENABLE_KVCACHE_CPU_OFFLOADING", "0") == "1":
        capacity = float(env.get("VLLM_TEST_SWAP_SPACE", "100"))
        if capacity <= 0:
            raise ValueError("The CPU KV capacity must be positive, in GiB")
        command.extend(
            ["--kv-offloading-size", str(capacity), "--kv-offloading-backend", "native"]
        )
    if env.get("VLLM_TEST_DISABLE_LOG_STATS", "0") == "1":
        raise ValueError("AE measurements require statistics; do not disable log stats")
    extra = shlex.split(env.get("VLLM_TEST_EXTRA_ARGS", ""))
    reserved = {
        "--additional-config",
        "--kv-offloading-size",
        "--kv-offloading-backend",
        "--speculative-config",
        "--disable-log-stats",
        "--scheduling-policy",
    }
    if any(arg.split("=", 1)[0] in reserved for arg in extra):
        raise ValueError(
            "Extra server arguments must not override the selected AE mode"
        )
    command.extend(extra)
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get(
        "VLLM_TEST_CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "0")
    )
    env["VLLM_USE_SIMPLE_KV_OFFLOAD"] = "0"
    command = build_command(env)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
