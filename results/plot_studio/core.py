"""Data layer + shared constants for the plot studio.

Loads results/*.json (one per model) and exposes the universes everything else
keys off: MODELS, QTS, METRICS, NS, DOC5, plus stable per-series colours/markers
and the corpus-size label helper. Also defines the studio's output paths — all
inside this folder, so nothing the studio writes ever lands elsewhere.
"""
import os
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # used only here, for the palette constants

CODE_DIR    = os.path.dirname(os.path.abspath(__file__))   # results/plot_studio
RESULTS_DIR = os.path.dirname(CODE_DIR)                    # results  (where the model JSONs live)
PDF_PATH    = os.path.join(CODE_DIR, "summary.pdf")
PROFILE_DIR = os.path.join(CODE_DIR, "profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)

# ---------------------------------------------------------------- load
data = {}
for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    data[d["name"].split("_n")[0]] = d

_first  = data[next(iter(data))]
QTS     = list(_first["query_types"])                 # broad, specific, exact, auto_sentence, auto_raw
KS      = list(_first["ks"])                          # 8, 40, 200, 1000
METRICS = [f"recall@{k}" for k in KS] + ["mrr"]       # 5 metrics, mrr last
NS      = sorted(int(n) for n in _first["results"])   # 17 corpus sizes, 0 .. 5_000_000
MODELS  = list(data)                                  # 8 models
DOC5    = [n for n in (0, 10_000, 100_000, 1_000_000, 5_000_000) if n in NS]
_QI     = {qt: i for i, qt in enumerate(QTS)}


def val(model, metric, qt, n):
    """Single scalar: one model, one metric, one query type, one corpus size."""
    return float(data[model]["results"][str(n)][metric][_QI[qt]])


# throughput (s / batch of 64 docs) from durations.txt -> docs/sec
DUR_S = {
    "Promptriever": 5.9833, "GritLM": 4.3712, "Qwen8b": 4.3282, "Snowflake_v2": 1.9111,
    "BGE_M3": 1.8977, "Nomic_Embed_v2": 1.3476, "Qwen0.6b": 0.9308, "Jina_v3": 0.8445,
}
DOCS_PER_SEC = {m: 64.0 / DUR_S[m] for m in MODELS if m in DUR_S}

# ---- stable per-model / per-metric colour + marker across every figure
_PAL   = plt.cm.tab10(np.linspace(0, 1, 10))
COLOR  = {m: _PAL[i] for i, m in enumerate(MODELS)}
MARKER = {m: "os^DvP*X"[i % 8] for i, m in enumerate(MODELS)}
# recall@k's: a muted, darker plasma slice (deep purple -> red-orange, no bright yellow);
# MRR is a bold green that stands out against the warm recalls
_KPAL  = plt.cm.plasma(np.linspace(0.05, 0.65, len(KS)))
MCOLOR = {f"recall@{k}": _KPAL[i] for i, k in enumerate(KS)}
MCOLOR["mrr"] = "#16c60c"
# Distinct categorical hues (ColorBrewer Set1) for query types: overlapping bands
# then blend into *recognisable* mixes instead of muddy darks.
_QSET  = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3", "#a65628", "#f781bf"]
QCOLOR = {qt: _QSET[i % len(_QSET)] for i, qt in enumerate(QTS)}

QT_LABEL = {"broad": "broad", "specific": "specific", "exact": "exact",
            "auto_sentence": "auto (sent)", "auto_raw": "auto (raw)"}
ME_LABEL = {me: ("MRR" if me == "mrr" else me.replace("recall@", "R@")) for me in METRICS}


def nlab(n):
    if n >= 1_000_000:
        return (f"{n / 1_000_000:.1f}".rstrip("0").rstrip(".")) + "m"
    return f"{n // 1000}k" if n >= 1000 else str(n)
