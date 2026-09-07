"""Shared comparison runner for Figures 9, 11, and 12."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae.common import (
    DEFAULT_SEED,
    ModelSpec,
    default_mode_specs,
    get_model_specs,
    now_stamp,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
    run_matrix,
    write_markdown_cases,
)


def parse_args(
    *,
    title: str,
    output_name: str,
    modes: tuple[str, ...],
    tasks: tuple[str, ...],
    qps_list: tuple[float, ...],
    num_list: tuple[int, ...],
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument(
        "--output-root", type=Path, default=Path("ae/results") / output_name
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model", default="qwen2.5-14b-instruct", choices=sorted(get_model_specs())
    )
    model_group.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--dataset", type=Path, default=Path("dataset/agentcodeclean_new.json")
    )
    parser.add_argument("--tasks", default=",".join(tasks))
    parser.add_argument("--qps-list", default=",".join(str(v) for v in qps_list))
    parser.add_argument("--num-list", default=",".join(str(v) for v in num_list))
    parser.add_argument(
        "--modes", default=",".join(modes), help="Comma-separated modes to run."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
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
    parser.add_argument("--port-base", type=int, default=8055)
    parser.add_argument("--server-ready-timeout", type=int, default=600)
    parser.add_argument("--case-timeout", type=int, default=2400)
    parser.add_argument("--extra-server-args", default="")
    parser.add_argument(
        "--temporal-selection-policy",
        choices=["best_fit", "first_fit", "priority_first"],
        default=None,
    )
    parser.add_argument("--transfer-bandwidth-gbps", type=float, default=None)
    parser.add_argument("--transfer-base-time-s", type=float, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the experiment plan without starting servers.",
    )
    return parser.parse_args()


def run_comparison(
    *,
    experiment: str,
    title: str,
    output_name: str,
    modes: tuple[str, ...],
    tasks: tuple[str, ...],
    qps_list: tuple[float, ...],
    num_list: tuple[int, ...],
) -> int:
    args = parse_args(
        title=title,
        output_name=output_name,
        modes=modes,
        tasks=tasks,
        qps_list=qps_list,
        num_list=num_list,
    )
    device_count = len(parse_csv_strings(args.cuda_devices))
    if not 1 <= args.tensor_parallel_size <= device_count:
        raise SystemExit("Tensor parallel size must fit the visible GPU count")
    model = (
        ModelSpec(args.model_path.name, str(args.model_path.expanduser().resolve()))
        if args.model_path
        else get_model_specs()[args.model]
    )
    server_args = f"--tensor-parallel-size {args.tensor_parallel_size}"
    if args.tensor_parallel_size > 1:
        server_args += " --disable-custom-all-reduce"
    if args.extra_server_args:
        server_args += " " + args.extra_server_args
    run_root = args.output_root / now_stamp()
    modes = default_mode_specs()
    mode_names = parse_csv_strings(args.modes)
    unknown_modes = sorted(set(mode_names) - set(modes))
    if unknown_modes:
        raise SystemExit(f"Unknown modes: {', '.join(unknown_modes)}")
    selected_modes = [modes[name] for name in mode_names]
    shared_env: list[tuple[str, str]] = []
    if args.temporal_selection_policy is not None:
        shared_env.append(
            ("VLLM_TEMPORAL_SELECTION_POLICY", args.temporal_selection_policy)
        )
    if args.transfer_bandwidth_gbps is not None:
        shared_env.append(
            ("SUNCAKE_TRANSFER_BANDWIDTH_GBPS", str(args.transfer_bandwidth_gbps))
        )
    if args.transfer_base_time_s is not None:
        shared_env.append(
            ("VLLM_MCP_TRANSFER_BASE_TIME_S", str(args.transfer_base_time_s))
        )
    if shared_env:
        selected_modes = [
            replace(mode, extra_env=mode.extra_env + tuple(shared_env))
            for mode in selected_modes
        ]
    cases, blocked = run_matrix(
        experiment=experiment,
        output_root=run_root,
        modes=selected_modes,
        tasks=parse_csv_strings(args.tasks),
        qps_list=parse_csv_floats(args.qps_list),
        num_list=parse_csv_ints(args.num_list),
        model=model,
        dataset=args.dataset,
        cuda_devices=args.cuda_devices,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        max_model_len=args.max_model_len,
        port_base=args.port_base,
        server_ready_timeout=args.server_ready_timeout,
        case_timeout=args.case_timeout,
        seed=args.seed,
        extra_server_args=server_args,
        dry_run=args.dry_run,
    )
    write_markdown_cases(run_root / "report.md", title, cases, blocked)
    print(run_root)
    return 0 if args.dry_run or (cases and not blocked) else 1
