# Data Provenance and Paper Alignment

The supplied `eurosys27-spring-paper407.pdf` is the reference for figure numbers,
captions, and published table values. The source results were copied before
processing. No file in `revision_result/` or `suncake_result/` was modified.

`data/manifest.json` records each source-relative path, source SHA-256, packaged
SHA-256, retained columns, row count, and processing description. These paths are
provenance identifiers; the artifact never reads the original result directories.
The plotting workflow uses processed data and does not claim to reproduce raw
measurements or every numerical statement in the paper by rerunning experiments.
The separately included implementation and workload inputs are described in
[SOURCE.md](SOURCE.md); they do not replace or regenerate these historical CSVs.

## Source Selection

| Paper item | Original source under the TokenCake project | Selection |
| --- | --- | --- |
| Figure 2(a) | `suncake_result/data/utlization_clean.csv` | Complete cleaned trace |
| Figure 3(a) | `suncake_result/data/contention_analysis.csv` | Complete cleaned preemption table |
| Figure 9 | `revision_result/data/latency/latency.csv` | Four visible systems and QPS 0.05, 0.1, 0.2, 0.5, 1.0 |
| Figure 10 | `suncake_result/data/gpu_usage.csv` | CodeWriter, Qwen2.5-14B, D2, SunCake/vLLM |
| Figure 11 | `revision_result/data/componment_ablation.csv` | Latency and throughput, both QPS levels |
| Figure 12 | `revision_result/data/mooncake_compare.csv` | Latency and throughput, both QPS levels |
| Figure 13 | `revision_result/data/parrot_compare.csv` | A100, 30-request runs, QPS 0.1, 0.2, 1.0, both applications |
| Table 4 | `revision_result/data/variability_summary.csv` | Published delta column, rounded to the table's one decimal |
| Figure 14 | `revision_result/data/temporal_req_choose.csv` | Three policies |
| Figure 15 | `revision_result/data/spatial_knobs.csv` | Three high watermarks |
| Figure 16 | `revision_result/data/time_tradeoff.csv` | Five cached-context lengths |

The three `suncake_result` inputs were checked against the included PDF. Figure
2(a)'s seven sampled annotations are 4.8, 4.5, 4.8, 4.7, 4.4, 18.5, and 5.1 percent.
Figure 3(a)'s final counts are 687 total, 177 inversion, and 510 normal events.
Figure 10's ten values match the printed bar annotations, including 85.7 percent
for SunCake at 0.1 QPS. These inputs do not need statistical correction for those
plots.

## Corrected Method Assignments

The PDF's Figures 11 and 12 agree with the historical CSVs, but their method/value
assignments conflict with the accompanying prose. The authors confirmed that the
AE should follow the prose in Sections 7.3 and 7.4. The corrections below apply
only to the packaged CSVs. Each source row's latency and throughput are kept
together, at their original precision, and assigned to the corrected method at
the same QPS. `offload_agent` is the complete SunCake system.

| Figure | QPS | AE method | Historical CSV method | Latency (s) | Throughput (req/s) |
| --- | --- | --- | --- | ---: | ---: |
| 11 | 0.2 | Baseline | `offload_agent` | 496.920284 | 0.009220693 |
| 11 | 0.2 | Agent | `offload` | 433.919710 | 0.010658697 |
| 11 | 0.2 | Offload | `agent` | 458.789798 | 0.009919051 |
| 11 | 0.2 | SunCake | `baseline` | 394.203055 | 0.011745446 |
| 11 | 0.5 | Baseline | `baseline` | 508.191066 | 0.009251983 |
| 11 | 0.5 | Agent | `offload_agent` | 476.291569 | 0.010001966 |
| 11 | 0.5 | Offload | `agent` | 496.299100 | 0.009667598 |
| 11 | 0.5 | SunCake | `offload` | 406.794019 | 0.011713501 |
| 12 | 0.2 | Baseline | `offload` | 697.187004 | 0.006869692 |
| 12 | 0.2 | Mooncake | `baseline` | 524.124745 | 0.008630196 |
| 12 | 0.2 | Offload | `offload_agent` | 576.820050 | 0.008211540 |
| 12 | 0.2 | SunCake | `mooncake` | 499.151415 | 0.009332509 |
| 12 | 0.5 | Baseline | `baseline` | 610.436601 | 0.007830736 |
| 12 | 0.5 | Mooncake | `mooncake` | 532.656551 | 0.008983378 |
| 12 | 0.5 | Offload | `offload_agent` | 551.569762 | 0.008750441 |
| 12 | 0.5 | SunCake | `offload` | 383.904266 | 0.012370849 |

`data/manifest.json` also records the QPS-specific old-to-new method mapping and
updated checksums. The included PDF remains unchanged, so its original Figure 11
and 12 bars differ from the corrected AE outputs. Offload event columns are not
included in these two minimal CSVs because those figures do not plot them and
the paper does not specify corrected event counts.

For Figure 11 at 0.5 QPS, the full-precision throughput gain is 26.605%, whereas
the prose reports 33.3%. The latter is obtained from the rounded bar annotations
0.012 and 0.009 req/s. The AE preserves the full-precision values and does not
adjust measurements to match a percentage calculated from rounded annotations.
Similarly, the full-precision latency reduction is 19.953% at that load, while
the prose reports 19.9%. README percentages are recomputed from the packaged
values, so small differences from the paper's printed percentages remain visible.

## Published Values and Limits

Table 4 contains the published deltas `-14.8`, `+8.3`, and `-3.4` percent. The old
summary also contains mean latencies that imply `+38.3%` at noise 0.25, inconsistent
with its stored delta and the paper. The package retains only the published table
values, not those inconsistent means or auxiliary counters. It does not infer new
mean latencies from the reported percentages.

Figure 9 uses the final processed grid, including its 72B data. The earlier raw
analysis script does not reconstruct that exact grid; it is not included as a
data-generation claim. At A100/14B CodeWriter D1, 1 QPS, the supplied plotted values
are 2085.716469 s (SunCake) and 2630.364058 s (vLLM), a 20.706% reduction. The paper's
47.06% prose claim is not recovered from those two points.

The Figure 13 source rows that match the PDF's numerical bars are labeled as
30-request runs, whereas its prose describes 20 applications. Figure 11's source
JSON metadata describes 7B/5-application runs, whereas its caption describes
14B/20 applications. The minimal CSVs retain the published plotting values; no
new experiment configuration is inferred from the captions.

## Plot Transformations

- Figure 2(a) preserves the original plot's selection of the highest sample-mean
  window of at most 1000 seconds, followed by its latter half by sample index.
  Agent block counts are summed directly. Percentages divide by 14266 blocks.
  The seven annotations are sampled points; 18.5% is not the maximum of every
  sample in the complete input trace.
- Figure 3(a) counts events cumulatively. An inversion has
  `victim_priority > preempt_priority`.
- Figure 9 places the five QPS values at categorical positions and marks compressed
  empty intervals on the y-axis where the original plot uses an axis break.
- Figure 10 retains the paper's 60-90 percent y-axis range and rounds displayed
  annotations to one decimal.
- Figure 13 follows the PDF's actual side-by-side panels, despite the caption's
  top/bottom wording, and preserves the marked axis breaks.
- Figure 14 uses separate axes for seconds and event counts. Figure 16 uses a
  logarithmic time axis; its CSV stores all times in milliseconds.

Displayed annotations may be rounded; stored values retain their available
precision except the explicitly rounded Table 4 deltas. Input CSVs are never
modified by plotting or data preparation.
