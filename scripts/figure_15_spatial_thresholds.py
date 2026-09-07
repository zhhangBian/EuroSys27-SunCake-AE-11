#!/usr/bin/env python3
"""Figure 15: Spatial pressure thresholds from the supplied CSV."""

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

from ae.plotting import _annotate, _bars


def figure_15(data_dir: Path, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    rows = sorted(read_rows(data_dir, "spatial_thresholds.csv"),
                  key=lambda row: float(row["high_watermark"]))
    labels = [f"{float(row['high_watermark']):.2f}" for row in rows]
    styles = [("#4C78A8", "///"), ("#54A24B", "\\\\\\"),
              ("#F58518", "xxx"), ("#B279A2", "---")]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    metrics = [("avg_app_latency_s", "Latency (s)", 1),
               ("throughput_rps", "Throughput (req/s)", 3)]
    for ax, (metric, ylabel, precision) in zip(axes, metrics):
        values = [float(row[metric]) for row in rows]
        for index, value in enumerate(values):
            color, hatch = styles[index % len(styles)]
            _bars(ax, [index * 0.82], [value], 0.56, color, hatch)
        _annotate(ax, [index * 0.82 for index in range(len(rows))], values,
                  precision, max(values) * 0.02, fontsize=11)
        ax.set_xticks([index * 0.82 for index in range(len(rows))], labels)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_ylim(0, max(values) * 1.18)
        ax.tick_params(labelsize=12)
        style_axes(ax)
    fig.supxlabel("High watermark", fontsize=15, y=0.025)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.96, wspace=0.34)
    return save_figure(fig, output_dir, "figure_15_spatial_thresholds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    paths = figure_15(args.data_dir, args.output_dir)
    print(json.dumps({"figure": "15", "files": [str(path.resolve()) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
