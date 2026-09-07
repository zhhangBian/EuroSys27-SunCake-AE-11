"""Drawing helpers shared by the individual SunCake figure scripts."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ae"
EXTERNAL_DATA_DIR = ROOT / "data" / "external"
OUTPUT_DIR = ROOT / "ae_outputs"


def read_rows(data_dir: Path, filename: str) -> list[dict[str, str]]:
    with (data_dir / filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def configure_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.8,
        "hatch.linewidth": 1.0,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 200,
    })


def style_axes(ax, integer_y: bool = False) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", color="grey", alpha=0.55)
    ax.tick_params(axis="both", direction="in", top=True, right=True, length=5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)
    if integer_y:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))


def save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.{extension}" for extension in ("png", "pdf")]
    fig.savefig(paths[0], bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white", metadata={"CreationDate": None})
    plt.close(fig)
    return paths


def _engine(value: str) -> str:
    return {
        "tokencake": "SunCake",
        "suncake": "SunCake",
        "mooncake": "Mooncake",
        "vllm": "vLLM",
        "vllm-prefix": "vLLM-Prefix",
    }.get(value.strip().lower().replace("_", "-"), value)


MODE_STYLES = {
    "baseline": ("Baseline", "#4C78A8", "///"),
    "agent": ("Agent", "#B279A2", "\\\\\\"),
    "mooncake": ("Mooncake", "#B279A2", "\\\\\\"),
    "offload": ("Offload", "#54A24B", "xxx"),
    "offload_agent": ("SunCake", "#F58518", "---"),
    "tokencake": ("SunCake", "#E17C05", "///"),
    "suncake": ("SunCake", "#E17C05", "///"),
    "parrot": ("Parrot", "#4C78A8", "\\\\\\"),
}


def _legend(styles: list[tuple[str, str, str]]) -> list[Patch]:
    return [
        Patch(facecolor="white", edgecolor=color, hatch=hatch,
              linewidth=1.6, label=label)
        for label, color, hatch in styles
    ]


def _bars(ax, positions, values, width, color, hatch):
    ax.bar(positions, values, width=width, facecolor="white",
           edgecolor=color, hatch=hatch, linewidth=0, zorder=2)
    return ax.bar(positions, values, width=width, facecolor="none",
                  edgecolor="black", linewidth=1.6, zorder=3)


def _annotate(ax, positions, values, precision, offset, fontsize=9):
    for position, value in zip(positions, values):
        ax.text(position, value + offset, f"{value:.{precision}f}",
                ha="center", va="bottom", fontsize=fontsize, zorder=4)


def _matrix(rows, identities, identity_key="mode"):
    grouped = {identity: {} for identity in identities}
    for row in rows:
        identity = row[identity_key]
        qps = float(row["request_rate"])
        grouped[identity][qps] = row
    rates = sorted({qps for values in grouped.values() for qps in values})
    return grouped, rates


def _comparison(data_dir, output_dir, filename, stem, modes, precisions):
    configure_matplotlib()
    grouped, rates = _matrix(read_rows(data_dir, filename), modes)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    metrics = [("avg_app_latency_s", "Average Latency (s)"),
               ("throughput_rps", "Throughput (req/s)")]
    width = 0.76 / len(modes)
    for ax, (metric, ylabel), precision in zip(axes, metrics, precisions):
        maximum = max(float(grouped[mode][qps][metric])
                      for mode in modes for qps in rates)
        offset = maximum * 0.018
        for index, mode in enumerate(modes):
            positions = [x + (index - (len(modes) - 1) / 2) * width
                         for x in range(len(rates))]
            values = [float(grouped[mode][qps][metric]) for qps in rates]
            _, color, hatch = MODE_STYLES[mode]
            _bars(ax, positions, values, width * 0.62, color, hatch)
            _annotate(ax, positions, values, precision, offset, fontsize=8)
        ax.set_xticks(range(len(rates)), [f"QPS {qps:g}" for qps in rates])
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(labelsize=13)
        ax.set_ylim(0, maximum * 1.22)
        style_axes(ax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    fig.legend(handles=_legend([MODE_STYLES[mode] for mode in modes]),
               loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=len(modes), frameon=False, fontsize=12,
               handlelength=1.15, handletextpad=0.3, columnspacing=0.65)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19,
                        top=0.825, wspace=0.33)
    return save_figure(fig, output_dir, stem)
