"""Averaging weights + weighted aggregations.

Weights come from sliders in [0,1]. With `normalize` on they are the weights
directly (renormalised to sum to 1). With it off the weights are softmax of the
slider logits, so a slider at 0/1 means probability 0/1. `_norm` additionally
renormalises over whatever subset of values is currently selected.
"""
import math

from core import METRICS, QTS, DOC5, val

# Defaults expressed as raw proportions (each group already sums to 1).
DEFAULT_METRIC_W = {me: (0.5 if me == "mrr" else 0.5 / (len(METRICS) - 1)) for me in METRICS}
DEFAULT_QUERY_W  = {qt: (0.25 / 2 if qt in ("auto_sentence", "auto_raw") else 0.25) for qt in QTS}
DEFAULT_DOC_W    = {n: 1.0 / len(DOC5) for n in DOC5}


def _norm(weights, keys):
    """Renormalise `weights` restricted to `keys` so they sum to 1 (equal fallback)."""
    keys = list(keys)
    if not keys:
        return {}
    sub = {k: max(0.0, float(weights.get(k, 0.0))) for k in keys}
    tot = sum(sub.values())
    if tot <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: v / tot for k, v in sub.items()}


def _logit(s):
    if s <= 0:
        return -math.inf
    if s >= 1:
        return math.inf
    return math.log(s / (1.0 - s))


def _softmax(zs):
    pos = [i for i, z in enumerate(zs) if z == math.inf]
    if pos:                                            # any +inf -> those share equally, rest 0
        return [1.0 / len(pos) if i in pos else 0.0 for i in range(len(zs))]
    finite = [z for z in zs if z != -math.inf]
    if not finite:                                     # all -inf -> equal fallback
        return [1.0 / len(zs)] * len(zs)
    m = max(finite)
    ex = [0.0 if z == -math.inf else math.exp(z - m) for z in zs]
    tot = sum(ex)
    return [e / tot for e in ex] if tot > 0 else [1.0 / len(zs)] * len(zs)


def weights_from_sliders(pos, normalize):
    """`pos`: list of slider positions in [0,1] -> list of weights that sum to 1."""
    n = len(pos)
    if n == 0:
        return []
    if normalize:
        s = sum(max(0.0, p) for p in pos)
        return [max(0.0, p) / s for p in pos] if s > 0 else [1.0 / n] * n
    return _softmax([_logit(p) for p in pos])


def recompute_weights(config):
    """Fill config['weights'] from config['slider_pos'] + config['normalize']."""
    for g in ("metrics", "queries", "docs"):
        keys = list(config["slider_pos"][g])
        w = weights_from_sliders([config["slider_pos"][g][k] for k in keys], config["normalize"])
        config["weights"][g] = {k: wi for k, wi in zip(keys, w)}


# ---- weighted aggregations (each returns a scalar for one model at one n) ----
def avg_metrics(model, qt, n, wm):
    return sum(w * val(model, me, qt, n) for me, w in wm.items())


def avg_queries(model, metric, n, wq):
    return sum(w * val(model, metric, qt, n) for qt, w in wq.items())


def avg_metrics_queries(model, n, wm, wq):
    return sum(wq[qt] * sum(wm[me] * val(model, me, qt, n) for me in wm) for qt in wq)
