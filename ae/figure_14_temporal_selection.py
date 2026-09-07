#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae.common import (
    DEFAULT_SEED,
    ModelSpec,
    ModeSpec,
    get_model_specs,
    now_stamp,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
    run_matrix,
    write_markdown_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Figure 14: temporal selection policy comparison."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ae/results/figure_14_temporal_selection"),
    )
    parser.add_argument(
        "--model", default="qwen2.5-14b-instruct", choices=sorted(get_model_specs())
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--dataset", type=Path, default=Path("dataset/agentcodeclean_new.json")
    )
    parser.add_argument("--tasks", default="code")
    parser.add_argument("--qps-list", default="1.0")
    parser.add_argument("--num-list", default="20")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--kv-offloading-size",
        "--swap-space",
        dest="swap_space",
        type=float,
        default=100,
        help="CPU KV capacity in GiB.",
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--port-base", type=int, default=8455)
    parser.add_argument("--server-ready-timeout", type=int, default=600)
    parser.add_argument("--case-timeout", type=int, default=2400)
    parser.add_argument("--extra-server-args", default="")
    parser.add_argument("--transfer-bandwidth-gbps", type=float, default=None)
    parser.add_argument("--agent-gpu-usage-high-watermark", type=float, default=None)
    parser.add_argument("--agent-gpu-usage-low-watermark", type=float, default=None)
    parser.add_argument("--reserve-adjustment-step", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.output_root / now_stamp()
    policies = ["best_fit", "first_fit", "priority_first"]
    base_env: list[tuple[str, str]] = []
    if args.transfer_bandwidth_gbps is not None:
        base_env.append(
            ("SUNCAKE_TRANSFER_BANDWIDTH_GBPS", str(args.transfer_bandwidth_gbps))
        )
    if args.agent_gpu_usage_high_watermark is not None:
        base_env.append(
            (
                "VLLM_AGENT_GPU_USAGE_HIGH_WATERMARK",
                str(args.agent_gpu_usage_high_watermark),
            )
        )
    if args.agent_gpu_usage_low_watermark is not None:
        base_env.append(
            (
                "VLLM_AGENT_GPU_USAGE_LOW_WATERMARK",
                str(args.agent_gpu_usage_low_watermark),
            )
        )
    if args.reserve_adjustment_step is not None:
        base_env.append(
            ("VLLM_AGENT_RESERVE_ADJUSTMENT_STEP", str(args.reserve_adjustment_step))
        )
    modes = [
        ModeSpec(
            name=f"offload_agent_{policy}",
            offloading_enabled=True,
            scheduling_policy="agent",
            enable_agent_scheduling=True,
            extra_env=tuple(base_env + [("VLLM_TEMPORAL_SELECTION_POLICY", policy)]),
        )
        for policy in policies
    ]
    cases, blocked = run_matrix(
        experiment="figure_14",
        output_root=run_root,
        modes=modes,
        tasks=parse_csv_strings(args.tasks),
        qps_list=parse_csv_floats(args.qps_list),
        num_list=parse_csv_ints(args.num_list),
        model=(
            ModelSpec(args.model_path.name, str(args.model_path.resolve()))
            if args.model_path
            else get_model_specs()[args.model]
        ),
        dataset=args.dataset,
        cuda_devices=args.cuda_devices,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        max_model_len=args.max_model_len,
        port_base=args.port_base,
        server_ready_timeout=args.server_ready_timeout,
        case_timeout=args.case_timeout,
        seed=args.seed,
        extra_server_args=args.extra_server_args,
        dry_run=args.dry_run,
    )
    write_markdown_cases(
        run_root / "report.md", "Figure 14: Temporal Selection", cases, blocked
    )
    print(run_root)
    return 0 if args.dry_run or (cases and not blocked) else 1


if __name__ == "__main__":
    raise SystemExit(main())
