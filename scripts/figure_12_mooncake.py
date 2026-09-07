#!/usr/bin/env python3
"""Figure 12: Mooncake comparison from the supplied CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ae.plotting import (
    EXTERNAL_DATA_DIR, OUTPUT_DIR, configure_matplotlib,
)

configure_matplotlib()

from ae.plotting import _comparison


def figure_12(data_dir: Path, output_dir: Path) -> list[Path]:
    return _comparison(data_dir, output_dir, "mooncake.csv",
                       "figure_12_mooncake",
                       ["baseline", "mooncake", "offload", "offload_agent"],
                       [0, 4])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=EXTERNAL_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    paths = figure_12(args.data_dir, args.output_dir)
    print(json.dumps({"figure": "12", "files": [str(path.resolve()) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
