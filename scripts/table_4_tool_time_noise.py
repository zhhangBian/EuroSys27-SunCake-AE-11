#!/usr/bin/env python3
"""Table 4: Tool-time prediction noise from the supplied CSV."""

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


def table_4(data_dir: Path, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    rows = sorted(read_rows(data_dir, "noise.csv"),
                  key=lambda row: float(row["noise_scale"]))
    scales = [f"{float(row['noise_scale']):g}" for row in rows]
    deltas = [f"{float(row['latency_delta_pct']):+.1f}" for row in rows]
    cells = [["Noise scale", *scales], ["Latency delta (%)", *deltas]]
    fig, ax = plt.subplots(figsize=(6.4, 1.35))
    ax.axis("off")
    table = ax.table(cellText=cells, loc="center", cellLoc="center",
                     colWidths=[0.43, 0.19, 0.19, 0.19], edges="horizontal")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.65)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        if row == 0:
            cell.get_text().set_fontweight("bold")
        if column == 0:
            cell.get_text().set_ha("left")
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.97)
    paths = save_figure(fig, output_dir, "table_4_noise")
    markdown = output_dir / "table_4_noise.md"
    markdown.write_text(
        "| Noise scale | " + " | ".join(scales) + " |\n"
        "| --- | ---: | ---: | ---: |\n"
        "| Latency delta (%) | " + " | ".join(deltas) + " |\n\n"
        "SunCake relative to agent-only scheduling; negative values mean lower latency.\n",
        encoding="utf-8",
    )
    return [*paths, markdown]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    paths = table_4(args.data_dir, args.output_dir)
    print(json.dumps({"figure": "table4", "files": [str(path.resolve()) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
