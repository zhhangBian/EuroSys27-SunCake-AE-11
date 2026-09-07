#!/usr/bin/env python3
"""Figure 14: Temporal request selection from the supplied CSV."""

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

from ae.plotting import _annotate, _bars, _legend


def figure_14(data_dir: Path, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    policies = ["offload_agent_best_fit", "offload_agent_first_fit",
                "offload_agent_priority_first"]
    rows = read_rows(data_dir, "temporal_selection.csv")
    by_policy = {row["mode"]: row for row in rows}
    fig, left = plt.subplots(figsize=(7.0, 4.0))
    right = left.twinx()
    metrics = [(left, "avg_app_latency_s", "Average Latency (s)", "Latency",
                "#4C78A8", "///", 1, -0.14),
               (right, "offload_events", "Offload Events", "Offload Events",
                "#F58518", "xxx", 0, 0.14)]
    for ax, metric, ylabel, _, color, hatch, precision, shift in metrics:
        values = [float(by_policy[policy][metric]) for policy in policies]
        positions = [index * 0.72 + shift for index in range(3)]
        _bars(ax, positions, values, 0.28, color, hatch)
        _annotate(ax, positions, values, precision, max(values) * 0.012,
                  fontsize=12)
        ax.set_ylim(0, max(values) * (1.16 if precision == 0 else 1.10))
        ax.set_ylabel(ylabel, color=color, fontsize=16)
        style_axes(ax, integer_y=precision == 0)
        ax.tick_params(axis="y", labelcolor=color, labelsize=13)
    right.grid(False)
    left.set_xticks([index * 0.72 for index in range(3)],
                    ["Best Fit", "First Fit", "Priority First"])
    left.tick_params(axis="x", labelsize=13)
    fig.legend(handles=_legend([(label, "black", hatch)
                                for _, _, _, label, _, hatch, _, _ in metrics]),
               loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2,
               frameon=False, fontsize=14)
    fig.subplots_adjust(left=0.14, right=0.86, bottom=0.18, top=0.81)
    return save_figure(fig, output_dir, "figure_14_temporal_selection")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    paths = figure_14(args.data_dir, args.output_dir)
    print(json.dumps({"figure": "14", "files": [str(path.resolve()) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
