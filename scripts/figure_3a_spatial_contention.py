#!/usr/bin/env python3
"""Figure 3(a): Spatial contention from the supplied CSV."""

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
from matplotlib.ticker import MaxNLocator


def figure_3a(data_dir: Path, output_dir: Path) -> list[Path]:
    rows = read_rows(data_dir, "spatial_contention.csv")
    groups: dict[str, list[float]] = {"Total": [], "Inversion": [], "Normal": []}
    for row in rows:
        timestamp = float(row["time"])
        groups["Total"].append(timestamp)
        kind = "Inversion" if int(row["victim_priority"]) > int(row["preempt_priority"]) else "Normal"
        groups[kind].append(timestamp)
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    for (label, times), color in zip(groups.items(), ("#377eb8", "#e41a1c", "#4daf4a")):
        ax.plot(sorted(times), range(1, len(times) + 1), label=label, color=color, linewidth=2.5)
    style_axes(ax, integer_y=True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Cumulative Count", fontsize=14)
    ax.set_title("Contention Analysis", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=13)
    fig.tight_layout()
    return save_figure(fig, output_dir, "figure_3a_spatial_contention")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    paths = figure_3a(args.data_dir, args.output_dir)
    print(json.dumps({"figure": "3a", "files": [str(path.resolve()) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
