#!/usr/bin/env python3
"""Provide the processed CSV for Table 4: Tool-time prediction noise."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "ae"


def main() -> int:
    if "--run" in sys.argv[1:]:
        sys.argv.remove("--run")
        from ae.table_4_tool_time_noise import main as run_experiment

        return run_experiment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--run", action="store_true",
        help="Run the retained GPU experiment; remaining arguments are passed through. Use --run --help.",
    )
    args = parser.parse_args()
    source = DATA_DIR / "noise.csv"
    destination = args.output_dir.resolve() / source.name
    if source.resolve() != destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
