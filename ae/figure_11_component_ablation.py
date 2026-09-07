#!/usr/bin/env python3

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae.comparison import run_comparison


def main() -> int:
    return run_comparison(
        experiment="figure_11",
        title="Figure 11: Component Ablation",
        output_name=Path(__file__).stem,
        modes=("baseline", "agent", "offload", "offload_agent"),
        tasks=("code",),
        qps_list=(0.2, 0.5),
        num_list=(20,),
    )


if __name__ == "__main__":
    raise SystemExit(main())
