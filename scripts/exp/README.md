# SunCake Experiment Entries

Each file corresponds to the same-named plotting script in `scripts/`.
By default it prints the bundled CSV path or copies the file using Python's
standard library. For figures with an existing GPU experiment,
`--run` invokes that retained implementation in `ae/`.

## Processed Data

Run an entry to locate its CSV, or copy it to a separate directory:

```bash
python scripts/exp/figure_9_e2e_latency.py
python scripts/exp/figure_9_e2e_latency.py --output-dir ae_outputs/data
python scripts/figure_9_e2e_latency.py --data-dir ae_outputs/data
```

`--output-dir` copies the CSV to that directory, replacing an existing file of
the same name. These entries provide the published plotting values.

| Paper item | Entry in `scripts/exp/` | CSV under `data/` | GPU execution with `--run` |
| --- | --- | --- | --- |
| Figure 2(a) | `figure_2a_temporal_utilization.py` | `ae/temporal_utilization.csv` | Supplied trace only |
| Figure 3(a) | `figure_3a_spatial_contention.py` | `ae/spatial_contention.csv` | Supplied trace only |
| Figure 9 | `figure_9_e2e_latency.py` | `ae/latency.csv` | End-to-end latency matrix |
| Figure 10 | `figure_10_gpu_utilization.py` | `ae/gpu_utilization.csv` | Supplied measurements only |
| Figure 11 | `figure_11_component_ablation.py` | `ae/ablation.csv` | Component ablation |
| Figure 12 | `figure_12_mooncake.py` | `external/mooncake.csv` | Requires Mooncake |
| Figure 13 | `figure_13_parrot.py` | `external/parrot.csv` | Requires Parrot |
| Figure 14 | `figure_14_temporal_selection.py` | `ae/temporal_selection.csv` | Temporal policy sweep |
| Figure 15 | `figure_15_spatial_thresholds.py` | `ae/spatial_thresholds.csv` | Spatial threshold sweep |
| Figure 16 | `figure_16_transfer.py` | `ae/transfer.csv` | Transfer microbenchmark |
| Table 4 | `table_4_tool_time_noise.py` | `ae/noise.csv` | Tool-time noise sweep |

## Minimal Main Experiment

Build the serving environment using [SOURCE.md](../../SOURCE.md), and run from
the artifact root:

```bash
.venv/bin/python scripts/exp/minimal_main.py \
  --model-path /path/to/Qwen2.5-14B-Instruct --cuda-devices 0 --dry-run

.venv/bin/python scripts/exp/minimal_main.py \
  --model-path /path/to/Qwen2.5-14B-Instruct --cuda-devices 0
```

The defaults select Figure 9's **QPS=1.0, CodeWriter D1, 20 applications** and
three modes: `vllm_vanilla` (vLLM), `baseline` (vLLM-Prefix), and `offload_agent`
(SunCake). Each case starts a fresh server. The original workload, generation
budgets, and tool waits are retained. This subset does not include the external
Mooncake case or the other Figure 9 panels.

Use one 80 GB NVIDIA GPU for the 14B default. SunCake reserves 16 GiB of host
memory for KV offloading, in addition to model-loading and process memory.
Budget **3-6 hours** after installation and model download; the
[timing breakdown](#minimal-figure-9-three-cases) explains the historical measurements and
planning allowances. The client timeout is 7,200 seconds per case and server
readiness allows another 600 seconds.
Adjust `--case-timeout`, `--kv-offloading-size`, and other retained Figure 9
arguments as needed. `--dry-run` writes a plan without starting a model.

Results are saved under `ae_outputs/runs/minimal_main/<timestamp>/`, including
`plan.json`, `cases.json`, `report.md`, per-case logs and summaries, and sampled
`metrics.jsonl`. Compare `avg_app_latency_s` across the three completed cases.
The updated runtime can produce different values from the historical CSVs.
Failed cases return a nonzero exit code and are recorded in `blocked.json`.

## Individual GPU Experiments

`--run` passes all remaining arguments to the existing experiment. Use the
serving environment, not the CPU plotting environment:

```bash
.venv/bin/python scripts/exp/figure_11_component_ablation.py --run --help
.venv/bin/python scripts/exp/figure_11_component_ablation.py --run \
  --model-path /path/to/Qwen2.5-14B-Instruct \
  --tasks code --qps-list 1.0 --num-list 20 \
  --kv-offloading-size 16 --case-timeout 7200 \
  --output-root ae_outputs/runs/ablation
```

These entries keep the existing experiment defaults, including full sweeps.
Without `--output-root`, their historical output location is
`ae/results/<experiment>/<timestamp>/`. They do not convert fresh JSON results
into the bundled paper CSVs. Figures 2(a), 3(a), and 10 only provide processed
data. Figure 16 times pinned CPU/GPU copies and a prefill with one output token.

Mooncake requires `MOONCAKE_REPO`, `CONDA_SH`, a `mooncake_agent` environment,
`mooncake_master`, and `server_mooncake.json`. Parrot requires `PARROT_REPO`,
`CONDA_SH`, a `parrot_cake` environment, and the engine/core configurations.
Their CSV preparation and plotting commands work without those installations.

### Setup and CPU Workflows

Reserve 5-10 minutes for the plotting environment and 1-4 hours for GPU
dependencies and a source build after the toolchains in [SOURCE.md](../../SOURCE.md)
are installed. These setup allowances are unmeasured and depend on network
speed, CPU parallelism, available RAM, and build caches. A slow source build
can take longer.

For downloads, use `time = total bytes / sustained bytes per second`. For
example, a 30 GiB payload takes 307 seconds at 100 MiB/s or 3,072 seconds at
10 MiB/s, before protocol overhead. This illustrates the calculation; use the
actual files required by the chosen model and environment.

Reserve one minute on CPU to produce all 11 PNG/PDF outputs. The minimal
experiment's `--dry-run` only writes plans and server commands and normally
finishes within seconds. Neither workflow needs model weights or a GPU.

### Minimal Figure 9: Three Cases

The inputs below are the A100/14B, CodeWriter D1, 20-application, QPS=1.0 rows
of [`latency.csv`](../../data/ae/latency.csv). Applications overlap in time:
QPS=1.0 describes their arrival rate, so roughly 20 seconds of arrivals can
still require hours to finish. Each application retains its full model calls
and tool waits.

| Mode | Historical average application latency | Planning allowance per complete case, including startup |
| --- | ---: | ---: |
| vLLM (`vllm_vanilla`) | 2,630.36 s = 43.84 min | About 68-120 min |
| vLLM-Prefix (`baseline`) | 2,373.02 s = 39.55 min | About 61-109 min |
| SunCake (`offload_agent`) | 2,085.72 s = 34.76 min | About 54-97 min |
| Three cases, sequential | 118.15 min = 1.97 h, sum of means | About 3.1-5.4 h; reserve 3-6 h |

A CSV mean is a reference for application latency; whole-case elapsed time
also includes arrival spacing, the slowest application's completion, and client
initialization. For scheduling purposes, use the following explicit heuristic,
with `L` the sum of historical mean latencies in minutes and `C` the case count:

```text
Lower planning allowance (minutes) = 1.5 * L + 2 * C
Upper planning allowance (minutes) = 2.5 * L + 10 * C
```

The 1.5-2.5 multipliers provide an unmeasured allowance for those workload
overheads; 2-10 minutes per case allows for server initialization, including
loading the model. These are schedule assumptions, not statistical confidence
limits or guarantees of completion. The same assumptions are used below for
the full Figure 9 estimate.

The minimal entry separately sets a **600-second server-readiness timeout**
and a **7,200-second client timeout** per case. Across three cases, these
configured waiting budgets sum to **6.5 hours**, plus launch, cleanup, and
reporting overhead. This is a cancellation budget, not a successful-run time
estimate or a strict overall deadline. A timed-out case is recorded as failed;
rerunning it requires additional time.

### Full Figure 9

Each model has `4 modes * 2 workloads * 5 QPS values * 2 application counts = 80`
cases. The paper's three model/hardware configurations therefore contain 240
cases. A single invocation of `figure_9_e2e_latency.py --run` selects one model;
it does not run all three configurations automatically.

| Paper configuration | Cases | Sum of historical mean latencies | Sequential planning allowance |
| --- | ---: | ---: | ---: |
| Qwen2.5-14B / A100 | 80 | 33.58 h | About 53-98 h |
| Qwen2.5-32B / H20 | 80 | 23.85 h | About 38-73 h |
| Qwen2.5-72B / H20 | 80 | 32.25 h | About 51-94 h |
| All three | 240 | 89.68 h | About 143-265 h, or 6-11 days |

The sums are computed from all 240 rows of `latency.csv`, without multiplying
each mean by the number of concurrent applications. The estimates include
240 separate startup allowances. They assume comparable hardware and workload
behavior; the updated runtime can differ. Larger models require their own
appropriate GPU allocation, model path, and tensor-parallel settings.

This covers Figure 9 only. Independent configurations can overlap in calendar
time when separate GPU allocations are available. Repeating each configuration
three times requires approximately three times the sequential allocation.
The full entry retains a 2,400-second client timeout, which is shorter than some
historical case means; choose an adequate `--case-timeout` when attempting the
full matrix. The minimal entry already raises it to 7,200 seconds.

### Additional Experiments

The following are the **current entry defaults and configured timeout budgets**.
These budgets help reserve a job slot but do not predict successful completion:

| Entry | Default case count and workload | Readiness + client budget, excluding launch/cleanup |
| --- | --- | --- |
| Figure 11 | 8: four modes, QPS 0.2/0.5, CodeWriter, 20 applications | `8 * (10 + 40)` min = 6 h 40 min |
| Figure 12 | 8: four modes including Mooncake, QPS 0.2/0.5, CodeWriter, 20 applications | `8 * (10 + 40)` min = 6 h 40 min; external setup is additional |
| Figure 13 | 1: Parrot, 7B, CodeWriter, QPS=1.0, 20 applications | Core 1 min + engine 4 min + probe 3 min + client 120 min = 2 h 8 min |
| Figure 14 | 3 policies, CodeWriter, QPS=1.0, 20 applications | `3 * (10 + 40)` min = 2 h 30 min |
| Figure 15 | 3 thresholds, CodeWriter, QPS=1.0, 20 applications | `3 * (10 + 40)` min = 2 h 30 min |
| Table 4 | 6: two modes, three noise scales, CodeWriter, QPS=0.5, 5 applications | `6 * (10 + 20)` min = 3 h |

Historical data do not establish completion estimates for every default:

- Figure 11's source metadata describes 7B/5-application runs, whereas the
  current default selects 14B/20 applications.
- Figure 13's CSV contains 12 rows: two workloads, three QPS values, and two
  backends, using 30 applications. Their mean latencies sum to 18.72 hours;
  the same planning heuristic gives about 29-49 hours for that comparison
  scope, separate from Figure 9. Its longest Parrot mean is 5.17 hours,
  already beyond the default two-hour client timeout. The default entry runs
  only its selected Parrot case; the SunCake cases require the local runner.
- Figures 14 and 15 contain measurements at QPS=3.0; their current defaults
  use QPS=1.0.
- Table 4 retains percentage changes only, so absolute elapsed time cannot
  be recovered from its CSV.

The defaults and timeout budgets above come from the retained `ae/` entries. For
an ancillary experiment on the current configuration, time one representative
completed case before allocating a larger sweep.

Figure 16 uses five context lengths. At each length, it runs 5 warmup + 30
measured transfer round trips and 2 warmup + 5 measured prefill calls. Applying
those counts to [`transfer.csv`](../../data/ae/transfer.csv) gives roughly
57 seconds of copy/prefill work, excluding allocations and model initialization.
Reserve **10-20 minutes** after the environment and weights are ready; this is
an unmeasured allowance, and initialization may take longer. Figures 2(a), 3(a),
and 10 have only the supplied-data workflow in this artifact.

### Measure Elapsed Time Locally

Use the shell's elapsed-time measurement for the complete experiment:

```bash
time .venv/bin/python scripts/exp/minimal_main.py \
  --model-path /path/to/Qwen2.5-14B-Instruct --cuda-devices 0
```

The shell's `real` time includes startup, execution, and cleanup. Per-case
`total_runtime_s` in `cases.json` measures the client process, while `dag_e2e_s`
measures the workload's arrival-to-final-completion interval. Use these actual
measurements to revise the allocation for the local machine. Additional seeds,
larger application counts, different context lengths, and altered KV capacity
can change both queueing and completion time.
