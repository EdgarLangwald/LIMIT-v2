"""Config object, signatures, profile (de)serialisation and PDF export.

The config is a plain dict: a `normalize` flag, per-group `slider_pos`, derived
`weights`, and a `plots` map of per-plot selections. Profiles are just this dict
written to JSON (sets become sorted lists; recomputed weights are re-derived on
load), saved under this folder's profiles/.
"""
import json

from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

from core import METRICS, QTS, MODELS, NS, DOC5, PDF_PATH
from weighting import DEFAULT_METRIC_W, DEFAULT_QUERY_W, DEFAULT_DOC_W, recompute_weights
from figures import BUILDERS, PLOT_ORDER


def default_config():
    cfg = {
        "normalize": True,
        "model_order": list(MODELS),          # custom display order, applied to every plot
        "slider_pos": {"metrics": dict(DEFAULT_METRIC_W),
                       "queries": dict(DEFAULT_QUERY_W),
                       "docs":    dict(DEFAULT_DOC_W)},
        "weights": {"metrics": {}, "queries": {}, "docs": {}},
        "plots": {
            "grid_a": {"show": True, "rows": set(METRICS), "cols": set(QTS), "lines": set(MODELS),
                       "xvals": set(NS), "grid": True, "mright": False, "mbottom": False, "corner": False},
            "grid_b": {"show": True, "rows": set(MODELS), "cols": set(QTS), "lines": set(METRICS),
                       "xvals": set(NS), "grid": True, "mright": False, "mbottom": False, "corner": False},
            "plot_c": {"show": True, "bands": set(QTS), "xvals": set(NS), "band_factor": 0.5},
            "grid_d": {"show": True, "rows": set(METRICS), "cols": set(DOC5), "hm_rows": set(MODELS),
                       "hm_cols": set(QTS), "grid": True, "mright": False, "mbottom": False, "corner": False},
            "pareto": {"show": True, "cols": set(DOC5), "models": set(MODELS), "grid": True, "mright": False},
        },
    }
    recompute_weights(cfg)
    return cfg


def plot_signature(key, config):
    """Hashable signature of everything that affects one plot's rendering."""
    c = config["plots"][key]; parts = [key]
    for field in sorted(c):
        v = c[field]
        parts.append((field, tuple(sorted(map(str, v))) if isinstance(v, set) else v))
    for name in ("metrics", "queries", "docs"):
        parts.append((name, tuple(sorted((str(k), round(x, 6)) for k, x in config["weights"][name].items()))))
    parts.append(("order", tuple(config["model_order"])))   # reordering invalidates the cache
    return repr(parts)


def config_to_json(config):
    out = {"normalize": config["normalize"],
           "model_order": list(config["model_order"]),
           "slider_pos": {g: {str(k): v for k, v in config["slider_pos"][g].items()} for g in config["slider_pos"]},
           "plots": {}}
    for key, c in config["plots"].items():
        out["plots"][key] = {f: (sorted(map(str, v)) if isinstance(v, set) else v) for f, v in c.items()}
    return out


def config_from_json(j):
    cfg = default_config()
    cfg["normalize"] = bool(j.get("normalize", True))
    saved = [m for m in j.get("model_order", []) if m in MODELS]
    cfg["model_order"] = saved + [m for m in MODELS if m not in saved]   # known order, then any new models

    for g in ("metrics", "queries", "docs"):
        for k, v in j.get("slider_pos", {}).get(g, {}).items():
            kk = int(k) if g == "docs" else k
            if kk in cfg["slider_pos"][g]:
                cfg["slider_pos"][g][kk] = float(v)
    for key, c in j.get("plots", {}).items():
        if key not in cfg["plots"]:
            continue
        for f, v in c.items():
            if f not in cfg["plots"][key]:
                continue
            if isinstance(cfg["plots"][key][f], set):
                want = {str(x) for x in v}
                cfg["plots"][key][f] = {x for x in cfg["plots"][key][f] if str(x) in want}
            else:
                cfg["plots"][key][f] = v
    recompute_weights(cfg)
    return cfg


def build_figures(config):
    """Return list of (key, Figure) for every shown plot that produced output."""
    w = config["weights"]; figs = []
    for key in PLOT_ORDER:
        c = config["plots"][key]
        if not c.get("show", True):
            continue
        fig = BUILDERS[key](c, w["metrics"], w["queries"], w["docs"], config["model_order"])
        if fig is not None:
            figs.append((key, fig))
    return figs


def export_pdf(config, path=PDF_PATH):
    figs = build_figures(config)
    with PdfPages(path) as pdf:
        if not figs:
            fig = Figure(figsize=(8, 4)); fig.text(0.5, 0.5, "Nothing selected", ha="center", va="center", fontsize=16)
            pdf.savefig(fig)
        for _key, fig in figs:
            pdf.savefig(fig)
    return path
