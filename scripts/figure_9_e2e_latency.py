#!/usr/bin/env python3
"""Figure 9: End-to-end latency from the supplied CSV.

Preserve the categorical QPS positions and marked compression of empty y-axis gaps."""

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

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ae.plotting import _engine


LATENCY_STYLES = {
    "SunCake": ("#ff7f00", "o"),
    "Mooncake": ("#4daf4a", "^"),
    "vLLM": ("#377eb8", "D"),
    "vLLM-Prefix": ("#a65628", "P"),
}


QPS_VALUES = (0.05, 0.1, 0.2, 0.5, 1.0)


def _axis_break(values: list[float]) -> tuple[float, float, float] | None:
    ordered = sorted(values)
    if len(ordered) < 8 or ordered[-1] <= ordered[0]:
        return None
    gap, index = max((ordered[i + 1] - ordered[i], i) for i in range(len(ordered) - 1))
    span = ordered[-1] - ordered[0]
    if index + 1 < 4 or len(ordered) - index - 1 < 4 or gap < 200 or gap < span * 0.35:
        return None
    low, high = ordered[index] + gap * 0.10, ordered[index + 1] - gap * 0.10
    compressed = min((high - low) * 0.35, max(span * 0.06, 1.0))
    return low, high, compressed


def _compress(value: float, axis_break: tuple[float, float, float] | None) -> float:
    if axis_break is None or value <= axis_break[0]:
        return value
    low, high, compressed = axis_break
    return value - (high - low - compressed)


def _mark_axis_break(ax, values: list[float], axis_break: tuple[float, float, float]) -> None:
    low, high, compressed = axis_break
    transformed = [_compress(value, axis_break) for value in values]
    padding = max((max(transformed) - min(transformed)) * 0.05, 1e-6)
    ax.set_ylim(min(transformed) - padding, max(transformed) + padding)
    ticks = []
    for value in [
        *MaxNLocator(nbins=3).tick_values(min(values), low),
        *MaxNLocator(nbins=3).tick_values(high, max(values)),
    ]:
        if min(values) <= value <= low or high <= value <= max(values):
            if not any(abs(value - existing) < 1e-9 for existing in ticks):
                ticks.append(value)
    ax.set_yticks([_compress(value, axis_break) for value in ticks])
    ax.set_yticklabels([f"{value:.0f}" if abs(value) >= 100 else f"{value:g}" for value in ticks])
    ax.axhspan(low, low + compressed, facecolor=ax.get_facecolor(), edgecolor="none", zorder=3)
    for boundary in (low, low + compressed):
        position = (boundary - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
        for side in (0, 1):
            ax.plot(
                (side - 0.012, side + 0.012), (position - 0.012, position + 0.012),
                transform=ax.transAxes, color="black", linewidth=2, clip_on=False, zorder=5,
            )


def figure_9(data_dir: Path, output_dir: Path) -> list[Path]:
    rows = read_rows(data_dir, "latency.csv")
    configure_matplotlib()
    fig, axes = plt.subplots(3, 4, figsize=(18.6, 10.65), squeeze=False)
    models = (("a100", "Qwen2.5-14B"), ("h20", "Qwen2.5-32B"), ("h20-72b", "Qwen2.5-72B"))
    cases = (("code", 20, "CodeWriter D1"), ("code", 30, "CodeWriter D2"),
             ("research", 20, "DeepResearch D1"), ("research", 30, "DeepResearch D2"))
    legend = {}
    for row_index, (device, model) in enumerate(models):
        for column_index, (task, count, title) in enumerate(cases):
            ax = axes[row_index][column_index]
            series: dict[str, dict[float, float]] = defaultdict(dict)
            for row in rows:
                engine = _engine(row["engine"])
                qps = float(row["qps"])
                if (row["device"] == device and row["task"] == task
                        and int(row["num_requests"]) == count
                        and engine in LATENCY_STYLES and qps in QPS_VALUES):
                    series[engine][qps] = float(row["avg_latency"])
            values = [value for points in series.values() for value in points.values()]
            axis_break = _axis_break(values)
            for engine, (color, marker) in LATENCY_STYLES.items():
                line, = ax.plot(
                    range(len(QPS_VALUES)), [_compress(series[engine][qps], axis_break) for qps in QPS_VALUES],
                    color=color, marker=marker, linewidth=2.8, markersize=11,
                    markeredgecolor="black", markeredgewidth=1.2, label=engine,
                )
                legend.setdefault(engine, line)
            style_axes(ax)
            ax.tick_params(labelsize=16)
            ax.set_xticks(range(len(QPS_VALUES)), [f"{qps:g}" for qps in QPS_VALUES])
            ax.margins(x=0.05)
            if axis_break:
                _mark_axis_break(ax, values, axis_break)
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.set_title(title, fontsize=18)
        axes[row_index][0].annotate(
            model, xy=(0, 0.5), xytext=(-64, 0), xycoords="axes fraction",
            textcoords="offset points", ha="center", va="center", rotation=90,
            fontsize=20, fontweight="bold",
        )
    fig.legend(
        list(legend.values()), list(legend), loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=4, frameon=False, fontsize=22, handletextpad=0.4, columnspacing=0.9, handlelength=1.2,
    )
    fig.supxlabel("QPS(application/s)", fontsize=20, y=0.045)
    fig.supylabel("Average Latency (s)", fontsize=20, x=0.015)
    fig.subplots_adjust(left=0.13, right=0.995, bottom=0.105, top=0.872, wspace=0.24, hspace=0.24)
    return save_figure(fig, output_dir, "figure_9_latency")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    paths = figure_9(args.data_dir, args.output_dir)
    print(json.dumps({"figure": "9", "files": [str(path.resolve()) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
