"""PDF report generation for LIMIT-v2 evaluation results.

Usage:
    from src.plot import visualize_results
    visualize_results()                          # reads all results/*.json
    visualize_results(simplify=True)             # x-axis in "pages" units
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.paths import RESULTS_DIR

MARKERS = "os^DvP*Xd"
COLORS  = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _fmt_x(n: int, simplify: bool) -> str:
    val = n
    unit = "pages" if simplify else "docs"
    if val == 0:
        return "0"
    if val >= 1000:
        return f"{val // 1000}k"
    return str(val)


def _x_label(simplify: bool) -> str:
    return "corpus size (pages)" if simplify else "n distractors"


def _load_results(results_dir: Path) -> dict[str, dict]:
    """Load all *.json files → {name: {n: metrics}}."""
    out = {}
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        name = data.get("name", p.stem)
        out[name] = {int(k): v for k, v in data["results"].items()}
    return out


def _page_per_model(pdf: PdfPages, name: str, results: dict[int, dict], simplify: bool) -> None:
    """One PDF page: all Recall@k curves overlaid on a single graph for one model."""
    ns = sorted(results)
    ks = sorted(int(k.split("@")[1]) for k in results[ns[0]] if k.startswith("recall@"))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(name, fontsize=14, fontweight="bold")

    for idx, k in enumerate(ks):
        vals = [results[n][f"recall@{k}"] for n in ns]
        ax.plot(ns, vals, marker=MARKERS[idx % len(MARKERS)],
                color=COLORS[idx % len(COLORS)], label=f"Recall@{k}")

    ax.set_xticks(ns)
    ax.set_xticklabels([_fmt_x(n, simplify) for n in ns], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel(_x_label(simplify), fontsize=10)
    ax.set_ylabel("Recall", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _page_mrr_comparison(pdf: PdfPages, all_results: dict[str, dict[int, dict]], simplify: bool) -> None:
    """One PDF page: MRR curves for all models overlaid."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("MRR — all models", fontsize=14, fontweight="bold")

    for i, (name, results) in enumerate(all_results.items()):
        ns = sorted(results)
        mrr = [results[n]["mrr"] for n in ns]
        ax.plot(ns, mrr, marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)], label=name)

    ns = sorted(next(iter(all_results.values())))
    ax.set_xticks(ns)
    ax.set_xticklabels([_fmt_x(n, simplify) for n in ns], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel(_x_label(simplify), fontsize=10)
    ax.set_ylabel("MRR", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def visualize_results(
    results_dir=None,
    output_pdf: Path | None = None,
    simplify: bool = False,
) -> Path:
    """Build a PDF report from all result JSONs in *results_dir*.

    Layout:
      - One page per model: all Recall@k curves overlaid on a single graph.
      - Final page: combined MRR comparison across all models.

    Args:
        results_dir: directory containing *.json result files (default: RESULTS_DIR).
        output_pdf:  output path for the PDF (default: <results_dir>/report.pdf).
        simplify:    if True, x-axis is labelled in "pages" units instead of
                     raw distractor counts — useful for presenter-facing reports.

    Returns the path to the written PDF.
    """
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    output_pdf  = Path(output_pdf)  if output_pdf  else results_dir / "report.pdf"

    all_results = _load_results(results_dir)
    if not all_results:
        raise FileNotFoundError(f"No result JSON files found in {results_dir}")

    with PdfPages(output_pdf) as pdf:
        for name, results in all_results.items():
            _page_per_model(pdf, name, results, simplify)
        _page_mrr_comparison(pdf, all_results, simplify)

    print(f"Saved: {output_pdf}  ({len(all_results)} model(s))")
    return output_pdf
