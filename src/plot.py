"""PDF report generation for LIMIT-v2 evaluation results.

Usage:
    from src.plot import visualize_results
    visualize_results()                          # reads all results/*.json
    visualize_results(simplify=True)             # x-axis in "pages" units
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.paths import RESULTS_DIR

MARKERS = "os^DvP*Xd"
COLORS  = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _fmt_x(n: int, simplify: bool) -> str:
    if n == 0:
        return "0"
    if n >= 1_000_000:
        val, suffix = n / 1_000_000, "m"
    elif n >= 1_000:
        val, suffix = n / 1_000, "k"
    else:
        return str(n)
    if val < 10:
        return f"{val:.2f}{suffix}"
    if val < 100:
        return f"{val:.1f}{suffix}"
    return f"{val:.0f}{suffix}"


def _x_label(simplify: bool) -> str:
    return "corpus size (pages)" if simplify else "n distractors"


def _load_results(
    results_dir: Path,
    n: int | None = None,
    model: str | None = None,
) -> dict[str, dict]:
    """Load *.json files → {name: {"query_types": [...], "results": {n: {metric: [vals]}}}}.

    Each metric value is a list indexed by the file's "query_types" legend. Legacy files
    (scalar metrics, no "query_types") are normalised to a single "all" type with 1-element
    lists, so callers can treat every file uniformly. Optionally filtered by n and/or model.
    """
    out = {}
    for p in sorted(results_dir.glob("*.json")):
        stem = p.stem
        if n is not None and f"_n{n}_" not in stem:
            continue
        if model is not None and model.lower() not in stem.lower():
            continue
        with open(p) as f:
            data = json.load(f)
        raw_name = data.get("name", stem)
        name     = re.sub(r"_n\d+.*$", "", raw_name)
        results  = {int(k): v for k, v in data["results"].items()}
        qts      = data.get("query_types")
        if qts is None:   # legacy scalar format → single "all" type
            qts     = ["all"]
            results = {nn: {mk: [mv] for mk, mv in metrics.items()}
                       for nn, metrics in results.items()}
        out[name] = {"query_types": qts, "results": results}
    return out


def _page_per_model(pdf: PdfPages, name: str, data: dict, qt: str, simplify: bool) -> None:
    """One PDF page: all Recall@k curves for one model, for query type *qt*."""
    results = data["results"]
    qi      = data["query_types"].index(qt)        # which slot in each metric list
    ns = sorted(results)
    ks = sorted(int(k.split("@")[1]) for k in results[ns[0]] if k.startswith("recall@"))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"{name}  —  {qt}", fontsize=14, fontweight="bold")

    for idx, k in enumerate(ks):
        vals = [results[n][f"recall@{k}"][qi] for n in ns]
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


def _page_mrr_comparison(pdf: PdfPages, models: dict[str, dict], qt: str, simplify: bool) -> None:
    """One PDF page: MRR curves for all models overlaid, for query type *qt*."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"MRR — all models  ({qt})", fontsize=14, fontweight="bold")

    for i, (name, data) in enumerate(models.items()):
        results = data["results"]
        qi      = data["query_types"].index(qt)
        ns      = sorted(results)
        mrr     = [results[n]["mrr"][qi] for n in ns]
        ax.plot(ns, mrr, marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)], label=name)

    ns = sorted(next(iter(models.values()))["results"])
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
    n: int | None = None,
    model: str | None = None,
    query_type: str | None = None,
) -> Path:
    """Build a PDF report from result JSONs in *results_dir*.

    Results are separated by query type. The report is organised into one section per query
    type (broad, specific, exact, auto_sentence, auto_raw), and within each section:
      - One page per model: all Recall@k curves overlaid on a single graph.
      - A combined MRR comparison across all models.

    Args:
        results_dir: directory containing *.json result files (default: RESULTS_DIR).
        output_pdf:  output path for the PDF (default: <results_dir>/report.pdf).
        simplify:    if True, x-axis is labelled in "pages" units instead of
                     raw distractor counts — useful for presenter-facing reports.
        n:           if given, only include JSONs whose filename contains ``_n{n}_``.
        model:       if given, only include JSONs whose filename contains this string
                     (case-insensitive). Both filters are applied as intersection.
        query_type:  if given, restrict the report to this single query type.

    Returns the path to the written PDF.
    """
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    output_pdf  = Path(output_pdf)  if output_pdf  else results_dir / "report.pdf"

    all_results = _load_results(results_dir, n=n, model=model)
    if not all_results:
        raise FileNotFoundError(f"No result JSON files found in {results_dir}")

    # Union of query types across files, preserving first-occurrence order.
    qt_order: list[str] = []
    for data in all_results.values():
        for qt in data["query_types"]:
            if qt not in qt_order:
                qt_order.append(qt)
    if query_type is not None:
        if query_type not in qt_order:
            raise ValueError(f"query_type {query_type!r} not found; available: {qt_order}")
        qt_order = [query_type]

    with PdfPages(output_pdf) as pdf:
        for qt in qt_order:
            models = {name: data for name, data in all_results.items()
                      if qt in data["query_types"]}
            if not models:
                continue
            for name, data in models.items():
                _page_per_model(pdf, name, data, qt, simplify)
            _page_mrr_comparison(pdf, models, qt, simplify)

    print(f"Saved: {output_pdf}  ({len(all_results)} model(s), {len(qt_order)} query type(s))")
    return output_pdf
