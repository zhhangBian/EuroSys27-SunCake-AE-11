#!/usr/bin/env python3
"""Figure 2(a): Temporal underutilization from the supplied CSV.

Preserve the original sample window, unscaled block sums, and 14266-block denominator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ae.plotting import (
    DATA_DIR, OUTPUT_DIR, configure_matplotlib,
    read_rows, save_figure, style_axes,
)

configure_matplotlib()

import matplotlib.pyplot as plt


AGENT_LABELS = {
    "mcp_conditional_execution": "Cond Execution",
    "mcp_external_test": "External Test",
    "mcp_external_tool_eval": "External Tool Eval",
    "mcp_file_query": "File Query",
    "mcp_file_write": "File Write",
    "mcp_search": "Search",
    "mcp_user_confirm": "User Confirm",
}


def _utilization_window(times: list[int], totals: list[int]) -> slice:
    if times[-1] - times[0] <= 1000:
        return slice(len(times) // 2, len(times))
    prefix = [0]
    for value in totals:
        prefix.append(prefix[-1] + value)
    best_average, best_start, best_end, end = -1.0, 0, len(times) - 1, 0
    for start, timestamp in enumerate(times):
        while end < len(times) and times[end] - timestamp <= 1000:
            end += 1
        average = (prefix[end] - prefix[start]) / (end - start)
        if average > best_average:
            best_average, best_start, best_end = average, start, end - 1
    return slice((best_start + best_end) // 2, best_end + 1)


def figure_2a(data_dir: Path, output_dir: Path) -> list[Path]:
    rows = read_rows(data_dir, "temporal_utilization.csv")
    agents = [key for key in rows[0] if key not in ("time", "gpu_kv_usage")]
    times = [int(row["time"]) for row in rows]
    counts = [[int(row[agent]) for row in rows] for agent in agents]
    totals = [sum(values) for values in zip(*counts)]
    window = _utilization_window(times, totals)
    times, totals = times[window], totals[window]
    counts = [values[window] for values in counts]

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    previous = [0] * len(times)
    for index, (agent, values) in enumerate(zip(agents, counts)):
        current = [a + b for a, b in zip(previous, values)]
        ax.fill_between(
            times, previous, current, color=plt.cm.tab20.colors[index % 20],
            alpha=0.5, label=AGENT_LABELS.get(agent, agent),
        )
        previous = current
    if len(times) > 1:
        for label_index in range(7):
            target = times[0] + label_index * (times[-1] - times[0]) / 6
            index = min(range(len(times)), key=lambda i: abs(times[i] - target))
            while index < len(times) and totals[index] == 0:
                index += 1
            if index == len(times):
                continue
            ax.scatter(times[index], totals[index], color="black", s=25, zorder=3)
            ax.annotate(
                f"{totals[index] / 14266 * 100:.1f}%",
                (times[index], totals[index]), xytext=(0, 2),
                textcoords="offset points",
                ha="left" if label_index == 0 else "right" if label_index == 6 else "center",
                va="bottom", fontsize=10,
            )
    style_axes(ax, integer_y=True)
    ax.set_title("Underutilized Block Count", fontsize=14)
    ax.set_xlabel("Elapsed Time (s)", fontsize=14)
    ax.set_ylabel("KV Cache Block Count", fontsize=14)
    ax.set_ylim(0, max(max(totals) * 1.10, 1))
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return save_figure(fig, output_dir, "figure_2a_temporal_utilization")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    paths = figure_2a(args.data_dir, args.output_dir)
    print(json.dumps({"figure": "2a", "files": [str(path.resolve()) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
