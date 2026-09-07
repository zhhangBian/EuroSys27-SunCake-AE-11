#!/usr/bin/env python3
"""Figure 16: Transfers and recomputation from the supplied CSV."""

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
from matplotlib.ticker import LogLocator, NullFormatter

from ae.plotting import _bars, _legend


def _milliseconds_label(value):
    if value >= 1000:
        return f"{value / 1000:.1f}s"
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def figure_16(data_dir: Path, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    rows = sorted(read_rows(data_dir, "transfer.csv"),
                  key=lambda row: int(row["cached_tokens"]))
    series = [("offload_ms", "D2H Offload", "#4C78A8", "///"),
              ("upload_ms", "H2D Upload", "#F58518", "xxx"),
              ("recompute_ms", "Recompute", "#54A24B", "\\\\\\")]
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    all_values = []
    for index, (metric, _, color, hatch) in enumerate(series):
        values = [float(row[metric]) for row in rows]
        all_values.extend(values)
        positions = [x + (index - 1) * 0.26 for x in range(len(rows))]
        _bars(ax, positions, values, 0.26, color, hatch)
        for x, value in zip(positions, values):
            ax.text(x, value * 1.12, _milliseconds_label(value),
                    ha="center", va="bottom", fontsize=10, zorder=4)
    style_axes(ax)
    ax.set_yscale("log")
    ax.set_ylim(max(min(all_values) * 0.45, 0.05), max(all_values) * 2.4)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(range(len(rows)), [row["cached_tokens"] for row in rows])
    ax.set_xlabel("Cached context length (tokens)", fontsize=17)
    ax.set_ylabel("Time (ms, log scale)", fontsize=17)
    ax.tick_params(labelsize=14)
    ax.tick_params(axis="y", which="minor", direction="in", right=True, length=3)
    fig.legend(handles=_legend([(label, color, hatch)
                                for _, label, color, hatch in series]),
               loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=3, frameon=False, fontsize=14)
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.16, top=0.85)
    return save_figure(fig, output_dir, "figure_16_transfer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    paths = figure_16(args.data_dir, args.output_dir)
    print(json.dumps({"figure": "16", "files": [str(path.resolve()) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
