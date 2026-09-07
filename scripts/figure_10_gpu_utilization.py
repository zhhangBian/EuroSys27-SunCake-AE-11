#!/usr/bin/env python3
"""Figure 10: GPU KV cache utilization from the supplied CSV.

Select 14B CodeWriter D2 and average duplicate engine/QPS rows."""

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

from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ae.plotting import _engine


def figure_10(data_dir: Path, output_dir: Path) -> list[Path]:
    rows = read_rows(data_dir, "gpu_utilization.csv")
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        engine = _engine(row["engine"])
        if (row["task"] == "CodeWriter" and row["model"] == "Qwen2.5-14B"
                and row["dataset"] == "D2" and engine in ("SunCake", "vLLM")):
            grouped[engine, float(row["qps"])].append(float(row["kv_cache_usage"]))
    qps_values = sorted({qps for _, qps in grouped})
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    handles = []
    for index, (engine, color, hatch) in enumerate((("SunCake", "#ff7f0e", "xxx"), ("vLLM", "#1f77b4", "///"))):
        values = [mean(grouped[engine, qps]) for qps in qps_values]
        positions = [position * 0.67 - 0.125 + index * 0.25 for position in range(len(qps_values))]
        ax.bar(positions, values, 0.25, facecolor="white", edgecolor=color, linewidth=0, hatch=hatch, zorder=2)
        ax.bar(positions, values, 0.25, facecolor="none", edgecolor="black", linewidth=2.2, zorder=3)
        for position, value in zip(positions, values):
            ax.text(position, value + 0.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=14)
        handles.append(Patch(facecolor="white", edgecolor=color, hatch=hatch, label=engine, linewidth=2))
    style_axes(ax)
    ax.set_ylim(60, 90)
    ax.set_xticks([index * 0.67 for index in range(len(qps_values))], [f"{qps:g}" for qps in qps_values])
    ax.set_xlabel("QPS(application/s)", fontsize=20)
    ax.set_ylabel("GPU Memory Usage (%)", fontsize=20)
    ax.tick_params(labelsize=17)
    ax.margins(x=0.03)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=20)
    fig.tight_layout()
    return save_figure(fig, output_dir, "figure_10_gpu_utilization")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    paths = figure_10(args.data_dir, args.output_dir)
    print(json.dumps({"figure": "10", "files": [str(path.resolve()) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
