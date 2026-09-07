#!/usr/bin/env python3
"""Run the retained Figure 9 experiment at QPS=1.0 for CodeWriter D1."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    arguments = [
        sys.executable,
        str(ROOT / "ae" / "figure_9_e2e_latency.py"),
        "--modes", "vllm_vanilla,baseline,offload_agent",
        "--tasks", "code",
        "--qps-list", "1.0",
        "--num-list", "20",
        "--kv-offloading-size", "16",
        "--case-timeout", "7200",
        "--output-root", str(ROOT / "ae_outputs" / "runs" / "minimal_main"),
        *sys.argv[1:],
    ]
    os.chdir(ROOT)
    os.execv(sys.executable, arguments)


if __name__ == "__main__":
    main()
