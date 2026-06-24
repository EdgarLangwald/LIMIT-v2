"""Local predictive power of sub-5M recall for recall@5M (per-column ICC shrinkage).

Question: to what degree does knowing a model's recall at a corpus size < 5M predict its
recall at the full 5M corpus -- and *where* (in which recall regime) is that prediction
locally reliable? Instead of fitting a parametric decay curve, at each corpus size we bin
experiments by their recall there, treat the bins as groups in a one-way random-effects
ANOVA whose response is the *true* recall@5M, and give each bin an empirical-Bayes
shrinkage estimate of recall@5M. The cell colour is a LOCAL reliability weight that uses
each bin's OWN within-bin spread of recall@5M (not just the global signal), so a bin whose
experiments agree -- and that has enough of them -- reads green; a noisy or sparse bin reds
out and its estimate is pulled to the grand mean.

  experiment   = (model, query_type, recall@k);  query_type in {broad, specific, exact, auto_raw}
                 (auto_sentence dropped: auto_* count as one). MRR never used.
  filter       = keep only experiments with recall >= 0.99 at D=0 distractors.
  x-axis       = corpus size N (every distractor count except 0 and the 5M target).
  y-axis       = NBINS recall bins, partition ADAPTIVE per column: interval [min, 1.0] where
                 min = lowest recall@N among the experiments, split into NBINS equal bins of
                 width h=(1-min)/NBINS. So every column's bins sit where its data actually is.
  response     = recall@5M.
  grand mean   = mu0 = mean recall@5M over the whole pool (constant across columns).
  per column   = one one-way ANOVA over the occupied bins -> between-bin variance tau^2,
                 pooled within variance sigma_pool^2 (=MS_within), and ICC rho.
  per bin i    = sigma_i^2 = bin's own within-variance of recall@5M, EB-shrunk toward
                 sigma_pool^2 with PRIOR_DF pseudo-df (so n_i=1 -> sigma_pool^2, not 0);
                 w_i = n_i*tau^2 / (n_i*tau^2 + sigma_i^2);  theta_i = w_i*xbar_i + (1-w_i)*mu0.
  heatmap      = colour is w_i (RdYlGn: green=locally reliable, red=shrunk to grand mean),
                 cell text is theta_i with n_i as a subscript.

Run from the repo root:  PYTHONPATH=. python results/plot_studio/icc_shrinkage.py
(core.py lives next to this file and is imported for the data layer.)
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# core.py sits in this same folder; ensure it is importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402  (data layer: MODELS, QTS, KS, NS, val, nlab, CODE_DIR)

QUERY_TYPES = ["broad", "specific", "exact", "auto_raw"]   # auto_sentence dropped
D0_THRESHOLD = 0.99                                        # >= this at D=0 to be kept
TARGET = max(core.NS)                                      # 5_000_000 -- the score we predict
NBINS = 8                                                  # per-column adaptive recall bins
PRIOR_DF = 2.0                                             # pseudo-df pulling sigma_i^2 -> sigma_pool^2


def build_pool(filter_d0=True):
    """Surviving experiments. Each: {model, qt, k, recall5m, by_cutoff: {N: recall@N}}.

    If filter_d0, keep only experiments with recall at D=0 >= D0_THRESHOLD; otherwise keep
    them all. The pool is identical for every column, so mu0 is constant across corpus sizes."""
    pool, dropped = [], 0
    for model in core.MODELS:
        for qt in QUERY_TYPES:
            for k in core.KS:
                metric = f"recall@{k}"
                if filter_d0 and core.val(model, metric, qt, 0) < D0_THRESHOLD:
                    dropped += 1
                    continue
                pool.append({
                    "model": model, "qt": qt, "k": k,
                    "recall5m": core.val(model, metric, qt, TARGET),
                    "by_cutoff": {N: core.val(model, metric, qt, N) for N in core.NS},
                })
    return pool, dropped


def adaptive_bins(recalls):
    """Per-column bin index of each recall@N value over [min, 1.0] split into NBINS bins.

    Returns (bands, lo, h). The experiment at the minimum lands in bin 0; recall==1.0
    clamps into the top bin. Degenerate column (all equal) -> everyone in the top bin."""
    recalls = np.asarray(recalls, float)
    lo = float(recalls.min())
    h = (1.0 - lo) / NBINS
    if h <= 0:
        return np.full(recalls.shape, NBINS - 1, int), lo, h
    bands = np.clip(((recalls - lo) / h).astype(int), 0, NBINS - 1)
    return bands, lo, h


def kendall_tau_b(x, y):
    """Kendall's tau-b rank correlation (tie-corrected). O(n^2), fine for the pool size."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    i, j = np.triu_indices(n, 1)
    sx = np.sign(x[i] - x[j])
    sy = np.sign(y[i] - y[j])
    p = sx * sy
    nc = int(np.sum(p > 0))
    nd = int(np.sum(p < 0))
    n0 = i.size
    n1 = int(np.sum(sx == 0))          # pairs tied in x
    n2 = int(np.sum(sy == 0))          # pairs tied in y
    denom = np.sqrt((n0 - n1) * (n0 - n2))
    return (nc - nd) / denom if denom > 0 else 0.0


def col_stats(responses, bands):
    """One-way random-effects ANOVA for a single corpus size, with per-bin local reliability.

    responses[j] = recall@5M, bands[j] = bin index of experiment j. mu0 is the global grand
    mean (passed in implicitly as responses.mean() == pool mean, constant). Returns
    (rho, per_bin) where per_bin[bin] = {n, xbar, w, theta}."""
    responses = np.asarray(responses, float)
    bands = np.asarray(bands, int)
    N_tot = responses.size
    mu0 = float(responses.mean())

    present = sorted(set(bands.tolist()))
    k_groups = len(present)
    counts = {g: int((bands == g).sum()) for g in present}
    means = {g: float(responses[bands == g].mean()) for g in present}
    ss = {g: float(((responses[bands == g] - means[g]) ** 2).sum()) for g in present}  # per-bin SS

    ssb = sum(counts[g] * (means[g] - mu0) ** 2 for g in present)
    ssw = sum(ss.values())
    df_b, df_w = k_groups - 1, N_tot - k_groups
    msb = ssb / df_b if df_b > 0 else 0.0
    sigma_pool2 = ssw / df_w if df_w > 0 else 0.0          # pooled within variance (prior scale)
    if k_groups > 1:
        n0 = (N_tot - sum(c * c for c in counts.values()) / N_tot) / (k_groups - 1)
    else:
        n0 = 0.0
    tau2 = max(0.0, (msb - sigma_pool2) / n0) if n0 > 0 else 0.0
    rho = tau2 / (tau2 + sigma_pool2) if (tau2 + sigma_pool2) > 0 else 0.0

    per_bin = {}
    for g in present:
        n_i = counts[g]
        # bin's own within-variance, EB-shrunk toward the pooled variance (scaled-inv-chi2 prior)
        sigma_i2 = (ss[g] + PRIOR_DF * sigma_pool2) / ((n_i - 1) + PRIOR_DF)
        denom = n_i * tau2 + sigma_i2
        w = (n_i * tau2) / denom if denom > 0 else 0.0
        theta = w * means[g] + (1.0 - w) * mu0
        per_bin[g] = {"n": n_i, "xbar": means[g], "w": w, "theta": theta}
    return rho, per_bin


def compute(filter_d0=True):
    pool, dropped = build_pool(filter_d0)
    if not pool:
        raise SystemExit("No experiments to analyse.")

    cols = [N for N in core.NS if N not in (0, TARGET)]
    responses = [e["recall5m"] for e in pool]
    mu0 = float(np.mean(responses))

    ncols = len(cols)
    W = np.full((NBINS, ncols), np.nan)
    THETA = np.full((NBINS, ncols), np.nan)
    Ncell = np.zeros((NBINS, ncols), int)
    rhos, taus, los = [], [], []

    for c, N in enumerate(cols):
        recalls_at_N = [e["by_cutoff"][N] for e in pool]
        bands, lo, _ = adaptive_bins(recalls_at_N)
        los.append(lo)
        taus.append(kendall_tau_b(recalls_at_N, responses))   # rank order at N -> order at 5M
        rho, per_bin = col_stats(responses, bands)
        rhos.append(rho)
        for g, s in per_bin.items():
            W[g, c] = s["w"]
            THETA[g, c] = s["theta"]
            Ncell[g, c] = s["n"]

    return dict(pool=pool, dropped=dropped, filter_d0=filter_d0, cols=cols, W=W, THETA=THETA,
                Ncell=Ncell, rhos=rhos, taus=taus, los=los, mu0=mu0)


def draw(ax, fig, res):
    """Render one heatmap panel onto ax (with its own colourbar) and set its title."""
    cols, W, THETA, Ncell, rhos, taus, los, mu0 = (res[k] for k in
                                                   ("cols", "W", "THETA", "Ncell", "rhos",
                                                    "taus", "los", "mu0"))
    ncols = len(cols)

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("#dddddd")                               # empty bins -> light grey
    im = ax.imshow(np.ma.masked_invalid(W), cmap=cmap, vmin=0.0, vmax=1.0,
                   aspect="auto", origin="lower")          # bin 0 (lowest recall) at the bottom

    # cell text: the reliability weight w_i (== the colour) with n as subscript
    for b in range(NBINS):
        for c in range(ncols):
            if np.isnan(W[b, c]):
                continue
            ax.text(c, b, rf"{W[b, c]:.2f}$_{{{Ncell[b, c]}}}$",
                    ha="center", va="center", fontsize=7.5, color="black")

    # x axis: corpus size + the per-column interval left edge (min) so bins are decodable
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([f"{core.nlab(N)}\nmin {lo:.2f}" for N, lo in zip(cols, los)], fontsize=7.5)
    ax.set_xlabel("corpus size used to predict recall@5M  (bottom line = min recall there)")

    # y axis: position within each column's [min, 1.0] interval (low -> high)
    ax.set_yticks(range(NBINS))
    ax.set_yticklabels([f"[{i/NBINS:.3g},{(i+1)/NBINS:.3g})" for i in range(NBINS)], fontsize=8)
    ax.set_ylabel("recall bin = position within column's [min, 1.0] interval  (low -> high)")

    # two per-column summary rows just above the grid: ICC rho and Kendall tau
    y_rho = NBINS - 0.5 + 0.4
    y_tau = NBINS - 0.5 + 1.0
    ax.text(-1.0, y_rho, r"ICC $\rho$", ha="right", va="center", fontsize=8, fontweight="bold")
    ax.text(-1.0, y_tau, r"Kendall $\tau$", ha="right", va="center", fontsize=8, fontweight="bold")
    for c in range(ncols):
        ax.text(c, y_rho, f"{rhos[c]:.2f}", ha="center", va="center", fontsize=7.5, color="#333333")
        ax.text(c, y_tau, f"{taus[c]:.2f}", ha="center", va="center", fontsize=7.5, color="#1a5fb4")
    ax.set_ylim(-0.5, NBINS - 0.5 + 1.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("local reliability $w_i$\n(green = bin's own recall@5M agree\n& enough data; red = shrunk to $\\mu_0$)",
                   fontsize=8)

    n_pool = len(res["pool"])
    if res["filter_d0"]:
        subset = f"{n_pool} experiments kept (recall@0$\\geq${D0_THRESHOLD:g}), {res['dropped']} dropped"
    else:
        subset = f"ALL {n_pool} experiments (no recall@0 filter)"
    ax.set_title(
        f"Local predictive power of sub-5M recall for recall@5M  (per-column ICC shrinkage)\n"
        f"{subset}   |   grand mean $\\mu_0$={mu0:.3f}   |   {NBINS} adaptive bins, prior df={PRIOR_DF:g}"
        f"   |   cell = $w_i$ (local reliability) = colour, subscript $n_i$",
        fontsize=8.5)


def render(results, out_base):
    """Stack one panel per result vertically into a single figure (.pdf + .png)."""
    ncols = len(results[0]["cols"])
    fig_w = ncols * 0.92 + 3.0
    fig_h = (NBINS * 0.66 + 2.6) * len(results)
    fig, axes = plt.subplots(len(results), 1, figsize=(fig_w, fig_h), squeeze=False)
    for ax, res in zip(axes[:, 0], results):
        draw(ax, fig, res)
    fig.tight_layout()

    pdf, png = out_base + ".pdf", out_base + ".png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return pdf, png


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(core.CODE_DIR, "icc_shrinkage"),
                    help="output path base (.pdf and .png appended)")
    args = ap.parse_args()

    results = [compute(filter_d0=True), compute(filter_d0=False)]   # top: filtered, bottom: all
    pdf, png = render(results, args.out)

    for res in results:
        tag = f"recall@0>={D0_THRESHOLD}" if res["filter_d0"] else "ALL (no filter)"
        print(f"=== {tag} ===")
        print(f"experiments : {len(res['pool'])} kept, {res['dropped']} dropped")
        print(f"grand mean mu0 (recall@5M over pool): {res['mu0']:.4f}")
        print("per-column (corpus size -> ICC rho, Kendall tau vs recall@5M, min recall there):")
        for N, rho, tau, lo in zip(res["cols"], res["rhos"], res["taus"], res["los"]):
            print(f"  {core.nlab(N):>6} : rho = {rho:.3f}   tau = {tau:.3f}   min = {lo:.3f}")
        print()
    print(f"wrote {pdf}\n      {png}")


if __name__ == "__main__":
    main()
