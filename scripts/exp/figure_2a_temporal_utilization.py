#!/usr/bin/env python3
"""Provide the processed CSV for Figure 2(a): Temporal underutilization."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "ae"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    source = DATA_DIR / "temporal_utilization.csv"
    destination = args.output_dir.resolve() / source.name
    if source.resolve() != destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
