#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae.common import (
    DEFAULT_SEED,
    get_model_specs,
    now_stamp,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
    run_matrix,
    write_json,
    write_markdown_cases,
    ModelSpec,
    ModeSpec,
)


def _noise_tag(scale: float) -> str:
    return f"{scale:g}".replace(".", "_")


def _mode_noise_scale(mode: str) -> float | None:
    if "_var_" not in mode:
        return None
    return float(mode.rsplit("_var_", 1)[1].replace("_", ".", 1))


def _mode_family(mode: str) -> str:
    if "_var_" not in mode:
        return mode
    return mode.rsplit("_var_", 1)[0]


def _client_noise_config(case: dict[str, Any]) -> dict[str, Any]:
    args = case.get("metadata", {}).get("mode_extra_client_args", [])
    config: dict[str, Any] = {}
    for index, item in enumerate(args):
        if item == "--tool_noise_profile" and index + 1 < len(args):
            config["profile"] = args[index + 1]
        elif item == "--tool_noise_scale" and index + 1 < len(args):
            config["scale"] = float(args[index + 1])
    return config


def _tool_noise_stats(app_json_path: str) -> dict[str, Any]:
    path = Path(app_json_path)
    if not path.exists():
        return {}
    app_info = json.loads(path.read_text(encoding="utf-8"))
    relative_errors: list[float] = []
    absolute_relative_errors: list[float] = []
    base_latencies: list[float] = []
    scheduled_latencies: list[float] = []
    type_counts: dict[str, int] = {}
    for app in app_info.values():
        for node in app.get("request_info", {}).values():
            if not node.get("is_mcp"):
                continue
            base = float(node.get("base_tool_latency", 0.0) or 0.0)
            scheduled = float(node.get("scheduled_tool_latency", 0.0) or 0.0)
            if base <= 0:
                continue
            relative_error = (scheduled - base) / base
            relative_errors.append(relative_error)
            absolute_relative_errors.append(abs(relative_error))
            base_latencies.append(base)
            scheduled_latencies.append(scheduled)
            node_type = str(node.get("type", "unknown") or "unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

    if not relative_errors:
        return {
            "mcp_call_count": 0,
            "tool_type_counts": type_counts,
        }
    return {
        "mcp_call_count": len(relative_errors),
        "avg_base_tool_latency_s": mean(base_latencies),
        "avg_scheduled_tool_latency_s": mean(scheduled_latencies),
        "mean_relative_error": mean(relative_errors),
        "std_relative_error": pstdev(relative_errors),
        "mean_absolute_relative_error": mean(absolute_relative_errors),
        "min_relative_error": min(relative_errors),
        "max_relative_error": max(relative_errors),
        "tool_type_counts": type_counts,
    }


def _aggregate_noise_cases(
    cases: list[Any], blocked: list[Any], run_root: Path
) -> dict[str, Any]:
    case_rows = [case if isinstance(case, dict) else case.__dict__ for case in cases]
    grouped: dict[str, dict[float, list[dict[str, Any]]]] = {
        "agent": {},
        "offload_agent": {},
    }
    for case in case_rows:
        noise = _mode_noise_scale(str(case.get("mode", "")))
        family = _mode_family(str(case.get("mode", "")))
        if noise is None or family not in grouped:
            continue
        row = dict(case)
        row["noise_scale"] = noise
        row["family"] = family
        row["client_noise_config"] = _client_noise_config(case)
        row["tool_noise_stats"] = _tool_noise_stats(str(case.get("app_json_path", "")))
        grouped[family].setdefault(noise, []).append(row)

    summary_rows = []
    all_noises = sorted(
        set(grouped["agent"].keys()) | set(grouped["offload_agent"].keys())
    )
    for noise in all_noises:
        agent_rows = grouped["agent"].get(noise, [])
        offload_rows = grouped["offload_agent"].get(noise, [])

        def avg(rows: list[dict[str, Any]], key: str) -> float | None:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            return mean(values) if values else None

        agent_latency = avg(agent_rows, "avg_app_latency_s")
        offload_latency = avg(offload_rows, "avg_app_latency_s")
        agent_throughput = avg(agent_rows, "throughput_rps")
        offload_throughput = avg(offload_rows, "throughput_rps")
        offload_decisions = int(
            sum(int(row.get("offload_decision_count", 0) or 0) for row in offload_rows)
        )
        offload_allowed = int(
            sum(int(row.get("offload_allowed_count", 0) or 0) for row in offload_rows)
        )
        offload_rejected = int(
            sum(int(row.get("offload_rejected_count", 0) or 0) for row in offload_rows)
        )
        offload_committed = int(
            sum(int(row.get("offload_events", 0) or 0) for row in offload_rows)
        )
        reject_rate = (
            offload_rejected / offload_decisions if offload_decisions > 0 else None
        )
        latency_delta_pct = None
        if agent_latency and offload_latency:
            latency_delta_pct = (
                (offload_latency - agent_latency) / agent_latency * 100.0
            )
        throughput_delta_pct = None
        if agent_throughput and offload_throughput:
            throughput_delta_pct = (
                (offload_throughput - agent_throughput) / agent_throughput * 100.0
            )
        reason_counts: dict[str, int] = {}
        for row in offload_rows:
            for reason, count in row.get("coordinator_reason_counts", {}).items():
                if not reason.startswith("offload_decision:"):
                    continue
                reason_counts[reason.replace("offload_decision:", "", 1)] = (
                    reason_counts.get(reason.replace("offload_decision:", "", 1), 0)
                    + int(count)
                )

        tool_stats_rows = [
            row.get("tool_noise_stats", {}) for row in agent_rows + offload_rows
        ]
        observed_abs_errors = [
            stats["mean_absolute_relative_error"]
            for stats in tool_stats_rows
            if stats.get("mean_absolute_relative_error") is not None
        ]

        summary_rows.append(
            {
                "noise_scale": noise,
                "agent_avg_latency_s": agent_latency,
                "offload_agent_avg_latency_s": offload_latency,
                "latency_delta_pct_offload_vs_agent": latency_delta_pct,
                "agent_throughput_rps": agent_throughput,
                "offload_agent_throughput_rps": offload_throughput,
                "throughput_delta_pct_offload_vs_agent": throughput_delta_pct,
                "offload_decision_count": offload_decisions,
                "offload_allowed_count": offload_allowed,
                "offload_rejected_count": offload_rejected,
                "offload_reject_rate": reject_rate,
                "offload_committed_events": offload_committed,
                "offload_reason_counts": reason_counts,
                "observed_mean_absolute_relative_tool_error": (
                    mean(observed_abs_errors) if observed_abs_errors else None
                ),
                "case_count": len(agent_rows) + len(offload_rows),
            }
        )

    return {
        "run_root": str(run_root),
        "blocked_count": len(blocked),
        "noise_summary": summary_rows,
        "notes": [
            "offload_decision_count is a sum of current TokenCake decision counters, including reevaluations.",
            "Each case retains its sampled /metrics counters in metrics.jsonl.",
            "The client keeps the base tool-time estimate fixed and adds noise to actual tool duration.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Table 4: tool-time prediction noise.")
    parser.add_argument(
        "--output-root", type=Path, default=Path("ae/results/table_4_tool_time_noise")
    )
    parser.add_argument(
        "--model", default="qwen2.5-14b-instruct", choices=sorted(get_model_specs())
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--dataset", type=Path, default=Path("dataset/agentcodeclean_new.json")
    )
    parser.add_argument("--tasks", default="code")
    parser.add_argument("--qps-list", default="0.5")
    parser.add_argument("--num-list", default="5")
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
    parser.add_argument("--port-base", type=int, default=8155)
    parser.add_argument("--server-ready-timeout", type=int, default=600)
    parser.add_argument("--case-timeout", type=int, default=1200)
    parser.add_argument("--noise-scales", default="0.0,0.25,0.5")
    parser.add_argument(
        "--noise-profile", choices=["gaussian", "uniform"], default="uniform"
    )
    parser.add_argument("--extra-server-args", default="")
    parser.add_argument("--transfer-bandwidth-gbps", type=float, default=None)
    parser.add_argument(
        "--temporal-selection-policy",
        choices=["best_fit", "first_fit", "priority_first"],
        default=None,
    )
    parser.add_argument("--agent-gpu-usage-high-watermark", type=float, default=None)
    parser.add_argument("--agent-gpu-usage-low-watermark", type=float, default=None)
    parser.add_argument("--reserve-adjustment-step", type=float, default=None)
    parser.add_argument("--max-new-tokens-cap", type=int, default=0)
    parser.add_argument("--smoke-max-finished-nodes", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.output_root / now_stamp()
    base_env: list[tuple[str, str]] = []
    if args.transfer_bandwidth_gbps is not None:
        base_env.append(
            ("SUNCAKE_TRANSFER_BANDWIDTH_GBPS", str(args.transfer_bandwidth_gbps))
        )
    if args.temporal_selection_policy is not None:
        base_env.append(
            ("VLLM_TEMPORAL_SELECTION_POLICY", args.temporal_selection_policy)
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
    modes: list[ModeSpec] = []
    for scale in parse_csv_floats(args.noise_scales):
        scale_tag = _noise_tag(scale)
        extra_client_args: tuple[str, ...] = (
            "--tool_noise_profile",
            args.noise_profile,
            "--tool_noise_scale",
            str(scale),
        )
        if args.max_new_tokens_cap > 0:
            extra_client_args += ("--max_new_tokens_cap", str(args.max_new_tokens_cap))
        if args.smoke_max_finished_nodes > 0:
            extra_client_args += (
                "--smoke_max_finished_nodes",
                str(args.smoke_max_finished_nodes),
            )
        modes.append(
            ModeSpec(
                name=f"agent_var_{scale_tag}",
                scheduling_policy="agent",
                enable_agent_scheduling=True,
                extra_env=tuple(base_env),
                extra_client_args=extra_client_args,
            )
        )
        modes.append(
            ModeSpec(
                name=f"offload_agent_var_{scale_tag}",
                scheduling_policy="agent",
                enable_agent_scheduling=True,
                offloading_enabled=True,
                extra_env=tuple(base_env),
                extra_client_args=extra_client_args,
            )
        )
    cases, blocked = run_matrix(
        experiment="table_4",
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
        run_root / "report.md", "Table 4: Tool-Time Noise", cases, blocked
    )
    analysis = _aggregate_noise_cases(cases, blocked, run_root)
    write_json(run_root / "table_4_noise_analysis.json", analysis)
    print(run_root)
    return 0 if args.dry_run or (cases and not blocked) else 1


if __name__ == "__main__":
    raise SystemExit(main())
