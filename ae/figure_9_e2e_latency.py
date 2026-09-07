#!/usr/bin/env python3

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae.comparison import run_comparison


def main() -> int:
    return run_comparison(
        experiment="figure_9",
        title="Figure 9: End-to-End Latency",
        output_name=Path(__file__).stem,
        modes=("vllm_vanilla", "baseline", "mooncake", "offload_agent"),
        tasks=("code", "research"),
        qps_list=(0.05, 0.1, 0.2, 0.5, 1.0),
        num_list=(20, 30),
    )


if __name__ == "__main__":
    raise SystemExit(main())
