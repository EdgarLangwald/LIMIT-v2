"""The hardcoded figure builders (one matplotlib Figure each).

Each builder is a pure function of its plot config slice `c`, the three weight
dicts and the custom model order, and returns a Figure (or None if the selection
is empty). The OO `Figure` API is used (no pyplot state) so figures can be built
off the main thread by the GUI.

  grid_a   "Grid A" 5x5 line grid : (recall@k's + mrr) x (query types), x = corpus size, model lines.
  grid_b   "Grid B" 8x5 line grid : (models) x (query types), x = corpus size, metric lines.
  plot_c   "Plot C" single bands  : per query type, mean +- factor*(min/max) of weighted-metric-avg.
  grid_d   "Grid D" 5x5 heatmaps  : (recall@k's + mrr) x (0,10k,100k,1m,5m docs); cell = model x query.
  pareto   "Pareto" 1-row grid    : per doc value, throughput vs weighted score, + Pareto frontier.

(The original Plot B was dropped, so the kinds were renumbered A/B/C/D — config
keys, builder names and display labels all use the new letters.)

Every score-vs-corpus-size plot shares one box aspect (CELL_BOX_ASPECT, taken
from a Grid A cell) so they all look the same shape — including the single
plots. Margins are four independent toggles separated from the body by extra
subplot padding plus a thick separator that spans the grid; the corner is its
own cell. A lone plot is centred at SINGLE_F of the page.
"""
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from core import (QTS, METRICS, NS, MODELS, DOC5, val, DOCS_PER_SEC,
                  COLOR, MARKER, MCOLOR, QCOLOR, QT_LABEL, ME_LABEL, nlab)
from weighting import _norm, avg_metrics, avg_queries, avg_metrics_queries

SINGLE_F        = 0.65   # a lone plot is centred at this fraction of the page width
CELL_BOX_ASPECT = 0.58   # measured height/width of a Grid A cell — given to single plots so they match


# ---------------------------------------------------------------- drawing helpers
def _fmt_x(ax):
    ax.set_xscale("linear")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: nlab(int(round(v))) if v >= 0 else ""))
    ax.tick_params(axis="x", labelsize=6, rotation=45)
    ax.tick_params(axis="y", labelsize=6)


def _line_cell(ax, xs, lines, colors, ylim=(0, 1.02)):
    for lab, y in lines.items():
        ax.plot(xs, y, lw=1.5, color=colors[lab], label=lab)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    _fmt_x(ax)   # cells fill their grid slot (no forced box aspect -> no whitespace gaps)


def _heat_cell(ax, M, row_labels, col_labels, show_x=True, show_y=True):
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels if show_x else [], rotation=40, ha="right", fontsize=5)
    ax.set_yticklabels(row_labels if show_y else [], fontsize=5)
    if M.size <= 60:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="black" if 0.3 < M[i, j] < 0.8 else "white", fontsize=4.5)
    return im


def _center(ax, f=SINGLE_F, box=True):
    """Shrink + centre a lone axis. With box=True, keep the shared score-vs-corpus aspect."""
    ax.set_position([(1 - f) / 2, (1 - f) / 2, f, f])
    if box:
        ax.set_box_aspect(CELL_BOX_ASPECT)


MARGIN_FLAGS = ("grid", "mright", "mbottom", "corner")


def _grid_dims(n_main_rows, n_main_cols, flags):
    has_main_rows = flags["grid"] or flags["mright"]
    has_bottom    = flags["mbottom"] or flags["corner"]
    has_main_cols = flags["grid"] or flags["mbottom"]
    has_right     = flags["mright"] or flags["corner"]
    R = n_main_rows if has_main_rows else 0
    C = n_main_cols if has_main_cols else 0
    return R, C, R + (1 if has_bottom else 0), C + (1 if has_right else 0)


def _region(ri, ci, R, C, flags):
    has_bottom = flags["mbottom"] or flags["corner"]
    has_right  = flags["mright"] or flags["corner"]
    is_b = has_bottom and ri == R
    is_r = has_right and ci == C
    if is_b and is_r:
        return "corner" if flags["corner"] else None
    if is_b:
        return "bottom" if flags["mbottom"] else None
    if is_r:
        return "right" if flags["mright"] else None
    return "main" if flags["grid"] else None


def _make_grid(figsize, R, C, flags, spacer=0.22, wspace=None, hspace=None):
    """Build a figure whose body and margins are separated by a fixed-width empty
    spacer track in the GridSpec. The body keeps normal cell spacing, the gap is
    deterministic and never cut off, and the separator line lands in the spacer.
    Returns (fig, axes) where axes[ri][ci] is None for cells with no content."""
    has_bottom = flags["mbottom"] or flags["corner"]
    has_right  = flags["mright"] or flags["corner"]
    col_spacer = has_right and C > 0
    row_spacer = has_bottom and R > 0
    wr = [1.0] * C + ([spacer] if col_spacer else []) + ([1.0] if has_right else [])
    hr = [1.0] * R + ([spacer] if row_spacer else []) + ([1.0] if has_bottom else [])
    fig = Figure(figsize=figsize)
    gs = fig.add_gridspec(len(hr), len(wr), width_ratios=wr, height_ratios=hr, wspace=wspace, hspace=hspace)
    gcol = lambda ci: ci if ci < C else C + (1 if col_spacer else 0)
    grow = lambda ri: ri if ri < R else R + (1 if row_spacer else 0)
    nr = R + (1 if has_bottom else 0)
    nc = C + (1 if has_right else 0)
    axes = [[None] * nc for _ in range(nr)]
    for ri in range(nr):
        for ci in range(nc):
            if _region(ri, ci, R, C, flags) is not None:
                axes[ri][ci] = fig.add_subplot(gs[grow(ri), gcol(ci)])
    return fig, axes


def _finish_margins(fig, axes, R, C, flags, col_dim, row_dim, lw=3.0):
    """Draw the thick separators (in the spacer) and label what each margin
    averages over — the column-margin label sits *above* its line. Only drawn
    when there is a body to separate from."""
    if not flags["grid"]:
        return
    has_bottom = flags["mbottom"] or flags["corner"]
    has_right  = flags["mright"] or flags["corner"]
    pos = lambda a: a.get_position()
    allax = [a for row in axes for a in row if a is not None]
    ps = [pos(a) for a in allax]
    gx0 = min(p.x0 for p in ps); gx1 = max(p.x1 for p in ps)
    gy0 = min(p.y0 for p in ps); gy1 = max(p.y1 for p in ps)
    body = [axes[ri][ci] for ri in range(len(axes)) for ci in range(len(axes[0]))
            if ri < R and ci < C and axes[ri][ci] is not None]
    if has_right and C > 0 and body:
        col = [axes[ri][C] for ri in range(len(axes)) if axes[ri][C] is not None]
        if col:
            x = (max(pos(a).x1 for a in body) + min(pos(a).x0 for a in col)) / 2
            fig.add_artist(Line2D([x, x], [gy0, gy1], transform=fig.transFigure, color="0.15", lw=lw))
            fig.text(x, min(gy1 + 0.012, 0.995), f"margin over {col_dim}", ha="center", va="bottom",
                     fontsize=7, style="italic", color="0.35")
    if has_bottom and R > 0 and body:
        rowm = [axes[R][ci] for ci in range(len(axes[0])) if axes[R][ci] is not None]
        if rowm:
            y = (min(pos(a).y0 for a in body) + max(pos(a).y1 for a in rowm)) / 2
            fig.add_artist(Line2D([gx0, gx1], [y, y], transform=fig.transFigure, color="0.15", lw=lw))
            fig.text(0.013, y, f"margin over {row_dim}", va="center", rotation=90,
                     fontsize=7, style="italic", color="0.35")


# ---------------------------------------------------------------- builders
def build_grid_a(c, wm_full, wq_full, wd_full, morder):
    rows  = [m for m in METRICS if m in c["rows"]]
    cols  = [q for q in QTS if q in c["cols"]]
    xs    = [n for n in NS if n in c["xvals"]]
    lines = [m for m in morder if m in c["lines"]]
    flags = {k: c[k] for k in MARGIN_FLAGS}
    if not (rows and cols and xs and lines and any(flags.values())):
        return None
    wm = _norm(wm_full, rows); wq = _norm(wq_full, cols)
    R, C, nr, nc = _grid_dims(len(rows), len(cols), flags)
    if nr == 0 or nc == 0:
        return None
    single = nr == 1 and nc == 1
    if single:
        fig = Figure(figsize=(6, 5.5)); axes = [[fig.subplots()]]
    else:
        fig, axes = _make_grid((max(4, 2.4 * nc + 1), max(3.2, 2.0 * nr + 1)), R, C, flags)
    for ri in range(nr):
        for ci in range(nc):
            ax = axes[ri][ci]
            if ax is None:
                continue
            reg = _region(ri, ci, R, C, flags)
            if reg == "corner":
                ld = {m: [avg_metrics_queries(m, n, wm, wq) for n in xs] for m in lines}; title = "avg metrics x avg queries"
            elif reg == "right":
                me = rows[ri]; ld = {m: [avg_queries(m, me, n, wq) for n in xs] for m in lines}; title = f"{ME_LABEL[me]} | avg queries"
            elif reg == "bottom":
                qt = cols[ci]; ld = {m: [avg_metrics(m, qt, n, wm) for n in xs] for m in lines}; title = f"avg metrics | {QT_LABEL[qt]}"
            else:
                me, qt = rows[ri], cols[ci]; ld = {m: [val(m, me, qt, n) for n in xs] for m in lines}; title = f"{ME_LABEL[me]} | {QT_LABEL[qt]}"
            _line_cell(ax, xs, ld, COLOR); ax.set_title(title, fontsize=7)   # lines only, no markers
    fig.suptitle("Grid A — metrics x query types, model lines vs corpus size", fontsize=11, y=0.995)
    if single:
        axes[0][0].legend(fontsize=6, loc="lower left"); _center(axes[0][0])
    else:
        handles = [Line2D([], [], color=COLOR[m], lw=1.8) for m in lines]
        fig.legend(handles, lines, loc="upper center", ncol=min(len(lines), 8), fontsize=7,
                   frameon=False, bbox_to_anchor=(0.5, 0.955))
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        _finish_margins(fig, axes, R, C, flags, "query types", "metrics")
    return fig


def build_grid_b(c, wm_full, wq_full, wd_full, morder):
    rows  = [m for m in morder if m in c["rows"]]
    cols  = [q for q in QTS if q in c["cols"]]
    xs    = [n for n in NS if n in c["xvals"]]
    lines = [me for me in METRICS if me in c["lines"]]
    flags = {k: c[k] for k in MARGIN_FLAGS}
    if not (rows and cols and xs and lines and any(flags.values())):
        return None
    wq = _norm(wq_full, cols); wmodel = {m: 1.0 / len(rows) for m in rows}
    R, C, nr, nc = _grid_dims(len(rows), len(cols), flags)
    if nr == 0 or nc == 0:
        return None
    single = nr == 1 and nc == 1
    if single:
        fig = Figure(figsize=(6, 5.5)); axes = [[fig.subplots()]]
    else:
        fig, axes = _make_grid((max(4, 2.4 * nc + 1), max(3.2, 2.0 * nr + 1)), R, C, flags)
    lab = {me: ME_LABEL[me] for me in lines}; col = {ME_LABEL[me]: MCOLOR[me] for me in lines}
    for ri in range(nr):
        for ci in range(nc):
            ax = axes[ri][ci]
            if ax is None:
                continue
            reg = _region(ri, ci, R, C, flags)
            if reg == "corner":
                ld = {me: [sum(wmodel[m] * avg_queries(m, me, n, wq) for m in rows) for n in xs] for me in lines}; title = "avg models x avg queries"
            elif reg == "right":
                m = rows[ri]; ld = {me: [avg_queries(m, me, n, wq) for n in xs] for me in lines}; title = f"{m} | avg queries"
            elif reg == "bottom":
                qt = cols[ci]; ld = {me: [sum(wmodel[m] * val(m, me, qt, n) for m in rows) for n in xs] for me in lines}; title = f"avg models | {QT_LABEL[qt]}"
            else:
                m, qt = rows[ri], cols[ci]; ld = {me: [val(m, me, qt, n) for n in xs] for me in lines}; title = f"{m} | {QT_LABEL[qt]}"
            _line_cell(ax, xs, {lab[me]: y for me, y in ld.items()}, col); ax.set_title(title, fontsize=6.5)
    fig.suptitle("Grid B — models x query types, metric lines vs corpus size", fontsize=11, y=0.995)
    if single:
        axes[0][0].legend(fontsize=6, loc="lower left"); _center(axes[0][0])
    else:
        handles = [Line2D([], [], color=MCOLOR[me], lw=2) for me in lines]
        fig.legend(handles, [ME_LABEL[me] for me in lines], loc="upper center", ncol=len(lines),
                   fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.955))
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        _finish_margins(fig, axes, R, C, flags, "query types", "models")
    return fig


def build_plot_c(c, wm_full, wq_full, wd_full, morder):
    xs     = [n for n in NS if n in c["xvals"]]
    bands  = [q for q in QTS if q in c["bands"]]
    factor = float(c.get("band_factor", 0.5))
    if not (xs and bands):
        return None
    wm = _norm(wm_full, METRICS)
    fig = Figure(figsize=(7, 6)); ax = fig.subplots()
    for qt in bands:
        mat  = np.array([[avg_metrics(m, qt, n, wm) for n in xs] for m in MODELS])
        mean = mat.mean(0); mn = mat.min(0); mx = mat.max(0)
        lo   = np.clip(mean - factor * (mean - mn), 0, 1)
        hi   = np.clip(mean + factor * (mx - mean), 0, 1)
        ax.fill_between(xs, lo, hi, color=QCOLOR[qt], alpha=0.13, linewidth=0)
        ax.plot(xs, mean, lw=2, color=QCOLOR[qt], label=QT_LABEL[qt])
    _fmt_x(ax); ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3)
    ax.set_ylabel("weighted-metric score"); ax.set_xlabel("corpus size  (target docs + distractors)")
    ax.set_title(f"Plot C — query-type bands (width = {factor:.2f}× min/max across models)", fontsize=10)
    ax.legend(fontsize=8, title="query type", loc="lower left")
    _center(ax)
    return fig


def build_grid_d(c, wm_full, wq_full, wd_full, morder):
    rows    = [m for m in METRICS if m in c["rows"]]
    cols    = [n for n in DOC5 if n in c["cols"]]
    hm_rows = [m for m in morder if m in c["hm_rows"]]
    hm_cols = [q for q in QTS if q in c["hm_cols"]]
    flags   = {k: c[k] for k in MARGIN_FLAGS}
    if not (rows and cols and hm_rows and hm_cols and any(flags.values())):
        return None
    wm = _norm(wm_full, rows); wd = _norm(wd_full, cols)
    R, C, nr, nc = _grid_dims(len(rows), len(cols), flags)
    if nr == 0 or nc == 0:
        return None
    single = nr == 1 and nc == 1
    ylabels = hm_rows; xlabels = [QT_LABEL[q] for q in hm_cols]
    if single:
        fig = Figure(figsize=(6, 5.5)); axes = [[fig.subplots()]]
    else:
        fig, axes = _make_grid((max(4, 2.1 * nc + 1.5), max(3.5, 1.9 * nr + 1)), R, C, flags, wspace=0.3, hspace=0.5)
    im = None
    for ri in range(nr):
        for ci in range(nc):
            ax = axes[ri][ci]
            if ax is None:
                continue
            reg = _region(ri, ci, R, C, flags)
            if reg == "corner":
                M = np.array([[sum(wd[n] * sum(wm[me] * val(m, me, q, n) for me in rows) for n in cols) for q in hm_cols] for m in hm_rows]); title = "avg metrics x avg docs"
            elif reg == "right":
                me = rows[ri]; M = np.array([[sum(wd[n] * val(m, me, q, n) for n in cols) for q in hm_cols] for m in hm_rows]); title = f"{ME_LABEL[me]} | avg docs"
            elif reg == "bottom":
                n = cols[ci]; M = np.array([[sum(wm[me] * val(m, me, q, n) for me in rows) for q in hm_cols] for m in hm_rows]); title = f"avg metrics | {nlab(n)}"
            else:
                me, n = rows[ri], cols[ci]; M = np.array([[val(m, me, q, n) for q in hm_cols] for m in hm_rows]); title = f"{ME_LABEL[me]} | {nlab(n)}"
            im = _heat_cell(ax, M, ylabels, xlabels, show_x=(ri == nr - 1), show_y=(ci == 0)); ax.set_title(title, fontsize=6.5)
    fig.suptitle("Grid D — metrics x corpus size; each cell = model x query-type heatmap", fontsize=11, y=0.995)
    if single:
        _center(axes[0][0], box=False)   # heatmap keeps its own aspect, just centred
    else:
        if im is not None:
            fig.colorbar(im, ax=[a for row in axes for a in row if a is not None],
                         fraction=0.015, pad=0.02, label="score")
        _finish_margins(fig, axes, R, C, flags, "corpus sizes", "metrics")
    return fig


def build_pareto(c, wm_full, wq_full, wd_full, morder):
    cols   = [n for n in DOC5 if n in c["cols"]]
    models = [m for m in morder if m in c["models"] and m in DOCS_PER_SEC]
    show_grid, show_avg = c["grid"], c["mright"]
    if not (cols and models and (show_grid or show_avg)):
        return None
    wm = _norm(wm_full, METRICS); wq = _norm(wq_full, QTS); wd = _norm(wd_full, cols)
    panels = []
    if show_grid:
        panels += [(lambda m, n=n: avg_metrics_queries(m, n, wm, wq), nlab(n) + " docs") for n in cols]
    if show_avg:
        panels.append((lambda m: sum(wd[n] * avg_metrics_queries(m, n, wm, wq) for n in cols), "avg over docs"))
    nc = len(panels)
    fig = Figure(figsize=(max(4, 3.0 * nc + 1), 3.6)); axes = fig.subplots(1, nc, squeeze=False, sharey=True)
    xs = {m: DOCS_PER_SEC[m] for m in models}
    xmax = 1.1 * max(xs.values())
    for ci, (score_of, title) in enumerate(panels):
        ax = axes[0][ci]; ys = {m: score_of(m) for m in models}
        for m in models:
            ax.scatter(xs[m], ys[m], s=70, color=COLOR[m], marker=MARKER[m], edgecolor="k", linewidth=0.5, zorder=3)
            ax.annotate(m, (xs[m], ys[m]), xytext=(4, 3), textcoords="offset points", fontsize=6)
        front, best = [], -1.0
        for m in sorted(models, key=lambda m: xs[m], reverse=True):
            if ys[m] > best:
                front.append(m); best = ys[m]
        ax.plot([xs[m] for m in front], [ys[m] for m in front], "--", color="gray", lw=1.3, zorder=2)
        ax.set_xlabel("throughput (docs/s)", fontsize=8); ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.02); ax.grid(True, alpha=0.3); ax.set_title(title, fontsize=8); ax.tick_params(labelsize=6)
    axes[0][0].set_ylabel("weighted score (queries x metrics)")
    fig.suptitle("Pareto — throughput vs score per corpus size (dashed = frontier)", fontsize=11, y=0.99)
    if nc == 1:
        axes[0][0].set_position([0.16, 0.16, 0.7, 0.68])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        if show_grid and show_avg:           # separate the avg-over-docs margin panel
            avg_i = len(cols)
            ps = [axes[0][ci].get_position() for ci in range(nc)]
            gy0 = min(p.y0 for p in ps); gy1 = max(p.y1 for p in ps)
            x = (axes[0][avg_i - 1].get_position().x1 + axes[0][avg_i].get_position().x0) / 2
            fig.add_artist(Line2D([x, x], [gy0, gy1], transform=fig.transFigure, color="0.15", lw=3.0))
            fig.text(x, min(gy1 + 0.015, 0.995), "margin over docs", ha="center", va="bottom",
                     fontsize=7, style="italic", color="0.35")
    return fig


BUILDERS = {"grid_a": build_grid_a, "grid_b": build_grid_b,
            "plot_c": build_plot_c, "grid_d": build_grid_d, "pareto": build_pareto}
PLOT_ORDER = ["grid_a", "grid_b", "plot_c", "grid_d", "pareto"]
PLOT_TITLE = {"grid_a": "Grid A  ·  metrics × queries (model lines)",
              "grid_b": "Grid B  ·  models × queries (metric lines)",
              "plot_c": "Plot C  ·  query-type bands",
              "grid_d": "Grid D  ·  metrics × docs heatmaps",
              "pareto": "Pareto  ·  throughput × score per doc value"}
