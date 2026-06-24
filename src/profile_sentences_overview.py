"""Overview of profile_sentences categories: frequency table and pool-size histogram.

Usage:
    python -m src.profile_sentences_overview
"""
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from src.paths import RESULTS_DIR
from src.profile_sentences import generate as _gen

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _collect():
    cats = [(name, getattr(_gen, name)) for name in vars(_gen) if not name.startswith("_")]
    return [(name, c.frequency, c.pool_size) for name, c in cats if c.frequency > 0]


def _page_table(pdf: "PdfPages", data: list[tuple[str, int, int]]) -> None:
    freq_counts = Counter(freq for _, freq, _ in data)
    freq_pool   = {}
    for _, freq, ps in data:
        freq_pool.setdefault(freq, []).append(ps)

    freqs = sorted(freq_counts)
    rows = []
    for f in freqs:
        count  = freq_counts[f]
        avg_ps = np.mean(freq_pool[f])
        rows.append([str(f), str(count), f"{avg_ps:.1f}"])

    total = sum(freq_counts.values())
    avg   = total / len(freq_counts)
    rows.append(["—", str(total), ""])
    rows.append(["—", f"{avg:.1f} avg", ""])

    col_labels = ["Frequency", "# Categories", "Avg pool size"]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("Profile-sentences: frequency distribution", fontsize=13, fontweight="bold")
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c5f8a")
            cell.set_text_props(color="white", fontweight="bold")
        elif r > len(freqs):
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f5f5")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _page_histogram(pdf: "PdfPages", data: list[tuple[str, int, int]]) -> None:
    data_sorted = sorted(data, key=lambda x: (x[1], x[0]))
    unique_freqs = sorted(set(f for _, f, _ in data_sorted))
    color_map = {f: COLORS[i % len(COLORS)] for i, f in enumerate(unique_freqs)}

    pool_sizes = [ps for _, _, ps in data_sorted]
    freqs_list = [f  for _, f, _ in data_sorted]
    bar_colors = [color_map[f] for f in freqs_list]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Pool size per category, grouped by frequency", fontsize=13, fontweight="bold")

    xs = np.arange(len(data_sorted))
    ax.bar(xs, pool_sizes, color=bar_colors, width=0.8)

    # Group boundary lines and x-axis frequency labels
    group_mids = []
    i = 0
    for freq in unique_freqs:
        group = [j for j, (_, f, _) in enumerate(data_sorted) if f == freq]
        mid = (group[0] + group[-1]) / 2
        group_mids.append((mid, freq))
        if group[0] > 0:
            ax.axvline(group[0] - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xticks([mid for mid, _ in group_mids])
    ax.set_xticklabels([f"freq={f}" for _, f in group_mids], fontsize=9)
    ax.set_ylabel("Pool size (# words / items)", fontsize=10)
    ax.set_xlim(-0.5, len(data_sorted) - 0.5)
    ax.grid(axis="y", alpha=0.3)

    legend_patches = [mpatches.Patch(color=color_map[f], label=f"freq={f}") for f in unique_freqs]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def profile_sentences_overview(output_pdf=None) -> None:
    from pathlib import Path
    output_pdf = Path(output_pdf) if output_pdf else RESULTS_DIR / "profile_sentences_overview.pdf"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = _collect()

    with PdfPages(output_pdf) as pdf:
        _page_table(pdf, data)
        _page_histogram(pdf, data)

    print(f"Saved: {output_pdf}  ({len(data)} categories, freq>0)")


if __name__ == "__main__":
    profile_sentences_overview()
