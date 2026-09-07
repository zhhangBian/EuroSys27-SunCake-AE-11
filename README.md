# SunCake EuroSys 2027 Artifact Evaluation

This repository accompanies **SunCake: A KV-Cache-centric Serving Framework for
LLM-based Multi-Agent Applications**, included as
[`eurosys27-spring-paper407.pdf`](eurosys27-spring-paper407.pdf). Figure numbers below follow the included SunCake paper.

The artifact provides processed plotting data, one script per figure, and the
SunCake implementation with its existing GPU experiments. The repository root
also includes the Mooncake and Parrot source versions used in the paper's
comparison experiments.

## Artifact Layout

- `eurosys27-spring-paper407.pdf`: the paper associated with this artifact.
- `data/ae/*.csv`: processed SunCake experiment data used for plotting.
- `data/external/*.csv`: the Mooncake and Parrot comparison data.
- `scripts/figure_*.py`, `scripts/table_4_tool_time_noise.py`: individual plotting scripts.
- `scripts/exp/`: corresponding data-preparation entries and the minimal main experiment; see its [README](scripts/exp/README.md).
- `ae/`, `agent/`, `dataset/`: retained experiment implementations, application clients, and workload inputs.
- `vllm/`, `csrc/`, `cmake/`, `rust/`: SunCake serving implementation and build sources.
- `Mooncake-TokenCake/`: bundled Mooncake source; see its [README](Mooncake-TokenCake/README.md).
- `Parrot-TokenCake/`: bundled Parrot source; see its [README](Parrot-TokenCake/README.md).

## Environment

Plotting requires Python 3.12, a CPU, and approximately 1 GB of RAM. With `uv` installed:

```bash
uv venv --python 3.12 .venv-plot
source .venv-plot/bin/activate
uv pip install -r requirements.txt
```

GPU experiments use a separate environment with this source tree built; follow
[SOURCE.md](SOURCE.md). Model weights are supplied separately.

## AE Workflow

Regenerate each paper figure directly from its included CSV:

```bash
python scripts/figure_2a_temporal_utilization.py
python scripts/figure_3a_spatial_contention.py
python scripts/figure_9_e2e_latency.py
python scripts/figure_10_gpu_utilization.py
python scripts/figure_11_component_ablation.py
python scripts/figure_12_mooncake.py
python scripts/figure_13_parrot.py
python scripts/figure_14_temporal_selection.py
python scripts/figure_15_spatial_thresholds.py
python scripts/figure_16_transfer.py
python scripts/table_4_tool_time_noise.py
```

Outputs are written to `ae_outputs/` as PNG/PDF files, plus a Markdown version
of Table 4. Each script accepts `--data-dir` and `--output-dir`.

## Figure Mapping

| Paper item | Experiment | Processed data | Plot script in `scripts/` |
| --- | --- | --- | --- |
| Figure 2(a) | Temporal underutilization | `ae/temporal_utilization.csv` | `figure_2a_temporal_utilization.py` |
| Figure 3(a) | Spatial contention | `ae/spatial_contention.csv` | `figure_3a_spatial_contention.py` |
| Figure 9 | End-to-end latency | `ae/latency.csv` | `figure_9_e2e_latency.py` |
| Figure 10 | GPU KV cache utilization | `ae/gpu_utilization.csv` | `figure_10_gpu_utilization.py` |
| Figure 11 | Component ablation | `ae/ablation.csv` | `figure_11_component_ablation.py` |
| Figure 12 | Mooncake comparison | `external/mooncake.csv` | `figure_12_mooncake.py` |
| Figure 13 | Parrot comparison | `external/parrot.csv` | `figure_13_parrot.py` |
| Figure 14 | Temporal request selection | `ae/temporal_selection.csv` | `figure_14_temporal_selection.py` |
| Figure 15 | Spatial pressure thresholds | `ae/spatial_thresholds.csv` | `figure_15_spatial_thresholds.py` |
| Figure 16 | Transfers and recomputation | `ae/transfer.csv` | `figure_16_transfer.py` |
| Table 4 | Tool-time prediction noise | `ae/noise.csv` | `table_4_tool_time_noise.py` |

Data paths are relative to `data/`. The remaining paper items are illustrations
or descriptive tables, available in the included PDF.

## Experiment Time

| Stage | Time to reserve | Scope and assumptions |
| --- | --- | --- |
| CPU environment | 5-10 minutes | Create the environment and install plotting dependencies; network-dependent |
| GPU environment and source build | 1-4 hours | One-time allowance with CUDA, C++ and Rust toolchains available; CPU resources and downloads can extend this |
| Model download | File size / effective download rate | Separate from build/run time; a 30 GiB payload takes about 5 minutes at 100 MiB/s or 51 minutes at 10 MiB/s |
| All 11 plot/table outputs | Under 1 minute | CPU plotting from the included CSVs |
| Minimal Figure 9 experiment | 3-6 hours | One 80 GB GPU, 14B CodeWriter D1, QPS=1.0, 20 applications, three sequential modes and server starts |
| Full Figure 9 matrix | About 6-11 days sequentially | 80 cases per model, 240 across the paper's 14B/32B/72B configurations; includes per-case startup allowances |

The minimal experiment's three historical mean latencies are 43.84, 39.55, and
34.76 minutes. Their sum is **1.97 hours**; the 3-6 hour budget additionally
allows for completion of the slowest applications, client overhead, and model
startup. Full Figure 9 has a mean-latency sum of **89.68 hours** before these
allowances. GPU allocations must suit each model; the single-GPU requirement
above applies to the minimal 14B experiment.

GPU/build times are planning allowances, not new measured runtimes. Full paper
evaluation also needs the remaining experiments and external-system setup;
repetitions and failed-case reruns add time. Because full reproduction
is lengthy, the artifact provides the **QPS=1.0 minimal main experiment**.

## Data Policy

The CSVs contain the processed values used for plotting; raw benchmark logs and
model weights are not included. Mooncake and Parrot plots use the supplied
external comparison data and do not require installing those systems.
The per-figure `scripts/exp/` entries provide the CSV paths or copy the files.
Fresh GPU runs write separate results and do not overwrite the paper data.

Figures 11 and 12 include the author-confirmed method-label corrections.

## SunCake Overview

SunCake combines agent/DAG-aware scheduling, dynamic KV reservations, and KV
preservation during tool waits. The implementation retains the internal name
`tokencake` for API compatibility.

## Running SunCake Directly

After building the GPU environment as described in [SOURCE.md](SOURCE.md):

```bash
.venv/bin/python scripts/exp/minimal_main.py \
  --model-path /path/to/Qwen2.5-14B-Instruct --cuda-devices 0
```

This runs Figure 9's CodeWriter D1 workload with 20 applications at QPS=1.0,
comparing vLLM, vLLM-Prefix, and SunCake. Plans, logs, and latency summaries go
to `ae_outputs/runs/minimal_main/`. See [scripts/exp/README.md](scripts/exp/README.md)
for dry runs, resource settings, and individual experiments.

## License

The project retains the Apache-2.0 license and source headers. See [LICENSE](LICENSE).
