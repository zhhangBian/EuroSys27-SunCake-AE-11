#!/usr/bin/env python3
"""Figure 13: Parrot comparison from the supplied CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ae.plotting import (
    EXTERNAL_DATA_DIR, OUTPUT_DIR, configure_matplotlib,
    read_rows, save_figure, style_axes,
)

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from ae.plotting import MODE_STYLES, _bars, _legend, _matrix


def _compressed_axis(ax, low, high, maximum):
    # Compress the empty interval while preserving the actual data coordinates.
    ratio = 0.055
    compressed_high = low + (high - low) * ratio

    def forward(values):
        values = np.asarray(values)
        return np.where(values <= low, values,
                        np.where(values < high, low + (values - low) * ratio,
                                 values - high + compressed_high))

    def inverse(values):
        values = np.asarray(values)
        return np.where(values <= low, values,
                        np.where(values < compressed_high,
                                 low + (values - low) / ratio,
                                 values - compressed_high + high))

    ax.set_yscale("function", functions=(forward, inverse))
    top = float(inverse(forward(maximum) * 1.22))
    ax.set_ylim(0, top)
    candidates = np.concatenate((MaxNLocator(nbins=3).tick_values(0, low),
                                 MaxNLocator(nbins=3).tick_values(high, maximum)))
    ticks = sorted({float(value) for value in candidates
                    if 0 <= value <= top and (value <= low or value >= high)})
    ax.set_yticks(ticks, [_latency_label(value) for value in ticks])
    for boundary in (low, high):
        y = float(forward(boundary) / forward(top))
        for x in (0, 1):
            ax.plot((x - 0.018, x + 0.018), (y - 0.018, y + 0.018),
                    transform=ax.transAxes, color="black", linewidth=1.8,
                    clip_on=False, zorder=6)
    return forward, inverse


def _latency_label(value):
    return f"{value / 1000:.1f}k" if value >= 1000 else f"{value:.0f}"


def figure_13(data_dir: Path, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    rows = read_rows(data_dir, "parrot.csv")
    backends = {row["backend"] for row in rows}
    cake_backends = backends & {"tokencake", "suncake"}
    modes = [next(iter(cake_backends)), "parrot"]
    tasks = [("code", "CodeWriter", (3300.0, 11500.0)),
             ("research", "DeepResearch", (850.0, 2800.0))]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
    for index, (ax, (task, label, y_break)) in enumerate(zip(axes, tasks)):
        grouped, rates = _matrix([row for row in rows if row["task"] == task],
                                  modes, "backend")
        values_by_mode = {mode: [float(grouped[mode][qps]["avg_app_latency_s"])
                                 for qps in rates] for mode in modes}
        all_values = [value for values in values_by_mode.values() for value in values]
        style_axes(ax)
        forward, inverse = _compressed_axis(ax, *y_break, max(all_values))
        offset = float(forward(max(all_values))) * 0.025
        for mode_index, mode in enumerate(modes):
            positions = [x + (mode_index - 0.5) * 0.35 for x in range(len(rates))]
            values = values_by_mode[mode]
            _, color, hatch = MODE_STYLES[mode]
            _bars(ax, positions, values, 0.35, color, hatch)
            for x, value in zip(positions, values):
                ax.text(x, float(inverse(forward(value) + offset)),
                        _latency_label(value), ha="center", va="bottom",
                        fontsize=11, zorder=5)
        ax.set_xticks(range(len(rates)), [f"{qps:g}" for qps in rates])
        ax.set_xlabel("QPS (applications/s)", fontsize=14)
        ax.set_title(label, fontsize=15, pad=8)
        ax.tick_params(labelsize=12)
        if index == 0:
            ax.set_ylabel("Average Latency (s)", fontsize=15)
        else:
            ax.yaxis.tick_right()
    fig.legend(handles=_legend([MODE_STYLES[mode] for mode in modes]),
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2,
               frameon=False, fontsize=14)
    fig.subplots_adjust(left=0.10, right=0.93, bottom=0.20, top=0.79, wspace=0.18)
    return save_figure(fig, output_dir, "figure_13_parrot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=EXTERNAL_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    paths = figure_13(args.data_dir, args.output_dir)
    print(json.dumps({"figure": "13", "files": [str(path.resolve()) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
