#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae.common import ModelSpec, get_model_specs, now_stamp, parse_csv_ints, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Figure 16: measure KV transfer versus recomputation time."
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("ae/results/figure_16_transfer")
    )
    parser.add_argument(
        "--model", default="qwen2.5-14b-instruct", choices=sorted(get_model_specs())
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--cached-token-counts", default="1024,2048,3072,4096,5120")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transfer-iterations", type=int, default=30)
    parser.add_argument("--transfer-warmup", type=int, default=5)
    parser.add_argument("--recompute-iterations", type=int, default=5)
    parser.add_argument("--recompute-warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--skip-recompute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    low = int(rank)
    high = min(len(ordered) - 1, low + 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def summarize_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "avg": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "avg": statistics.mean(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def derive_block_shape(
    model_path: str, block_size: int, dtype_name: str
) -> dict[str, Any]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))
    num_attn_heads = getattr(
        config, "num_attention_heads", getattr(config, "n_head", None)
    )
    hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", None))
    num_kv_heads = getattr(config, "num_key_value_heads", num_attn_heads)
    head_dim = getattr(config, "head_dim", None) or hidden_size // num_attn_heads
    dtype_bytes = torch.empty((), dtype=torch_dtype(dtype_name)).element_size()
    block_bytes = block_size * num_layers * num_kv_heads * head_dim * 2 * dtype_bytes
    return {
        "source": "transformers_config",
        "num_layers": int(num_layers),
        "num_attention_heads": int(num_attn_heads),
        "num_key_value_heads": int(num_kv_heads),
        "hidden_size": int(hidden_size),
        "head_dim": int(head_dim),
        "dtype_bytes": int(dtype_bytes),
        "block_bytes": int(block_bytes),
    }


def measure_transfer(
    total_bytes: int, device: str, iterations: int, warmup: int
) -> dict[str, Any]:
    cpu_buf = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    gpu_buf = torch.empty(total_bytes, dtype=torch.uint8, device=device)
    torch.cuda.synchronize(device)

    h2d_ms: list[float] = []
    d2h_ms: list[float] = []
    for index in range(warmup + iterations):
        start = time.perf_counter()
        gpu_buf.copy_(cpu_buf, non_blocking=True)
        torch.cuda.synchronize(device)
        h2d_elapsed = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        cpu_buf.copy_(gpu_buf, non_blocking=True)
        torch.cuda.synchronize(device)
        d2h_elapsed = (time.perf_counter() - start) * 1000.0

        if index >= warmup:
            h2d_ms.append(h2d_elapsed)
            d2h_ms.append(d2h_elapsed)

    del cpu_buf
    del gpu_buf
    torch.cuda.empty_cache()
    return {
        "cpu_to_gpu_ms": summarize_ms(h2d_ms),
        "gpu_to_cpu_ms": summarize_ms(d2h_ms),
        "round_trip_ms": summarize_ms([h + d for h, d in zip(h2d_ms, d2h_ms)]),
    }


def build_prompt(rng: np.random.Generator, token_count: int) -> dict[str, list[int]]:
    prompt_token_ids = rng.integers(100, 10000, size=token_count).tolist()
    return {"prompt_token_ids": prompt_token_ids}


def measure_recompute(
    model_path: str,
    token_counts: list[int],
    dtype_name: str,
    iterations: int,
    warmup: int,
    seed: int,
    enforce_eager: bool,
    disable_chunked_prefill: bool,
) -> dict[int, dict[str, Any]]:
    from vllm import LLM, SamplingParams

    max_context = max(token_counts)
    llm = LLM(
        model=model_path,
        dtype=dtype_name,
        trust_remote_code=True,
        max_model_len=max_context + 1,
        max_num_batched_tokens=max_context + 1,
        max_num_seqs=1,
        gpu_memory_utilization=0.75,
        enable_prefix_caching=False,
        enforce_eager=enforce_eager,
        enable_chunked_prefill=not disable_chunked_prefill,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=1,
        detokenize=False,
    )
    rng = np.random.default_rng(seed)
    result: dict[int, dict[str, Any]] = {}
    for token_count in token_counts:
        elapsed_ms: list[float] = []
        for index in range(warmup + iterations):
            prompt = build_prompt(rng, token_count)
            start = time.perf_counter()
            llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
            latency_ms = (time.perf_counter() - start) * 1000.0
            if index >= warmup:
                elapsed_ms.append(latency_ms)
        result[token_count] = {
            "recompute_path_ms": summarize_ms(elapsed_ms),
            "raw_recompute_path_ms": elapsed_ms,
        }

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    run_root = args.output_root / now_stamp()
    run_root.mkdir(parents=True, exist_ok=True)

    model = (
        ModelSpec(args.model_path.name, str(args.model_path.resolve()))
        if args.model_path
        else get_model_specs()[args.model]
    )
    token_counts = parse_csv_ints(args.cached_token_counts)
    if args.dry_run:
        write_json(
            run_root / "plan.json",
            {
                "experiment": "figure_16",
                "model_path": model.path,
                "cached_token_counts": token_counts,
                "dtype": args.dtype,
                "block_size": args.block_size,
                "dry_run": True,
            },
        )
        print(run_root)
        return 0
    torch.cuda.set_device(args.device)
    block_shape = derive_block_shape(model.path, args.block_size, args.dtype)
    rows: list[dict[str, Any]] = []

    transfer_by_tokens: dict[int, dict[str, Any]] = {}
    device_props = torch.cuda.get_device_properties(args.device)
    if not args.skip_transfer:
        for token_count in token_counts:
            kv_block_count = math.ceil(token_count / args.block_size)
            total_bytes = int(block_shape["block_bytes"]) * kv_block_count
            transfer_by_tokens[token_count] = measure_transfer(
                total_bytes,
                args.device,
                args.transfer_iterations,
                args.transfer_warmup,
            )

    recompute_by_tokens: dict[int, dict[str, Any]] = {}
    if not args.skip_recompute:
        recompute_by_tokens = measure_recompute(
            model.path,
            token_counts,
            args.dtype,
            args.recompute_iterations,
            args.recompute_warmup,
            args.seed,
            args.enforce_eager,
            args.disable_chunked_prefill,
        )

    for token_count in token_counts:
        kv_block_count = math.ceil(token_count / args.block_size)
        total_bytes = int(block_shape["block_bytes"]) * kv_block_count
        row: dict[str, Any] = {
            "cached_tokens": token_count,
            "kv_block_count": kv_block_count,
            "block_size_tokens": args.block_size,
            "bytes_per_kv_block": int(block_shape["block_bytes"]),
            "total_kv_bytes": total_bytes,
        }
        row.update(transfer_by_tokens.get(token_count, {}))
        row.update(recompute_by_tokens.get(token_count, {}))
        rows.append(row)

    payload = {
        "experiment": "figure_16",
        "model_name": model.name,
        "model_path": model.path,
        "measurement_config": {
            "cached_token_counts": token_counts,
            "block_size": args.block_size,
            "dtype": args.dtype,
            "device": args.device,
            "device_name": device_props.name,
            "device_total_memory": device_props.total_memory,
            "transfer_iterations": args.transfer_iterations,
            "transfer_warmup": args.transfer_warmup,
            "recompute_iterations": args.recompute_iterations,
            "recompute_warmup": args.recompute_warmup,
            "seed": args.seed,
            "enforce_eager": args.enforce_eager,
            "enable_chunked_prefill": not args.disable_chunked_prefill,
            "derived_block_shape": block_shape,
        },
        "notes": [
            "cached_tokens is the x-axis; kv_block_count is ceil(cached_tokens / block_size).",
            "gpu_to_cpu_ms is offload time; cpu_to_gpu_ms is upload time.",
            "recompute_path_ms measures an uncached prefill of the specified token count plus one output token.",
        ],
        "rows": rows,
    }
    write_json(run_root / "summary.json", payload)
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
