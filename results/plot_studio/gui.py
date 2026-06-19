"""Tkinter control panel + detachable, auto-refreshing PDF preview.

Rendering runs on a background thread (figures use the OO API in `figures`), so
ticking boxes and scrolling stay responsive; an in-flight render is interrupted
the moment anything changes. Each plot is rendered and cached independently, so
a change only re-renders the plot it affects.
"""
import io
import os
import copy
import glob
import json
import time
import queue
import threading

from core import (QTS, METRICS, MODELS, NS, DOC5, nlab, ME_LABEL, QT_LABEL,
                  PDF_PATH, PROFILE_DIR)
from weighting import recompute_weights
from figures import BUILDERS, PLOT_ORDER, PLOT_TITLE
from state import default_config, plot_signature, config_to_json, config_from_json, export_pdf

RENDER_W = 2200   # px width each plot is rasterised at (preview downscales from this)


def launch_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    import fitz
    from PIL import Image, ImageTk

    config = default_config()
    cache = {}

    root = tk.Tk()
    root.title("LIMIT-v2 — plot studio")
    root.geometry("500x820")

    # ---- light theming ----
    BG, CARD, ACCENT, INK = "#f4f5f7", "#ffffff", "#3b6ea5", "#222"
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    root.configure(bg=BG)
    style.configure(".", background=BG, foreground=INK, font=("Segoe UI", 9))
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=CARD)
    style.configure("TLabel", background=BG)
    style.configure("Card.TLabel", background=CARD)
    style.configure("Hint.TLabel", background=BG, foreground="#777", font=("Segoe UI", 8))
    style.configure("TCheckbutton", background=CARD)
    style.configure("TLabelframe", background=CARD, borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", background=CARD, foreground=ACCENT, font=("Segoe UI", 9, "bold"))
    style.configure("TButton", padding=4)
    style.configure("Accent.TButton", foreground="#fff", background=ACCENT, padding=5)
    style.map("Accent.TButton", background=[("active", "#2f5a87")])
    style.configure("TScale", background=CARD)

    # ---- detachable preview window ----
    preview = tk.Toplevel(root)
    preview.title("LIMIT-v2 — preview")
    preview.geometry("980x920")
    preview.configure(bg="#3a3a3a")
    pv_canvas = tk.Canvas(preview, background="#3a3a3a", highlightthickness=0)
    pv_scroll = ttk.Scrollbar(preview, orient="vertical", command=pv_canvas.yview)
    pv_canvas.configure(yscrollcommand=pv_scroll.set)
    pv_scroll.pack(side="right", fill="y")
    pv_canvas.pack(side="left", fill="both", expand=True)
    preview.protocol("WM_DELETE_WINDOW", preview.withdraw)
    _pv_photos = []
    last_images = []
    status = tk.StringVar(value="starting…")

    # ---- background render plumbing ----
    state = {"gen": 0, "dirty": True, "running": False, "last": 0.0}
    result_q = queue.Queue()

    def render_plot(snap, key):
        sig = plot_signature(key, snap)
        ent = cache.get(key)
        if ent is not None and ent["sig"] == sig:
            return ent
        w = snap["weights"]
        fig = BUILDERS[key](snap["plots"][key], w["metrics"], w["queries"], w["docs"], snap["model_order"])
        if fig is None:
            ent = {"sig": sig, "pdf": None, "img": None}
        else:
            buf = io.BytesIO(); fig.savefig(buf, format="pdf")
            pdf_bytes = buf.getvalue()
            d = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = d[0]; zoom = RENDER_W / page.rect.width
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            d.close()
            ent = {"sig": sig, "pdf": pdf_bytes, "img": img}
        cache[key] = ent
        return ent

    def worker(g, snap):
        try:
            merged = fitz.open(); imgs = []
            for key in [k for k in PLOT_ORDER if snap["plots"][k]["show"]]:
                if g != state["gen"]:
                    break
                ent = render_plot(snap, key)
                if ent["pdf"]:
                    d = fitz.open(stream=ent["pdf"], filetype="pdf"); merged.insert_pdf(d); d.close()
                    imgs.append((key, ent["img"]))
            if g == state["gen"]:
                if merged.page_count == 0:
                    merged.new_page(width=600, height=300)
                merged.save(PDF_PATH)
            merged.close()
            result_q.put((g, imgs))
        except Exception as e:                          # surface to status line
            result_q.put((g, e))

    def start_worker():
        state["running"] = True
        threading.Thread(target=worker, args=(state["gen"], copy.deepcopy(config)), daemon=True).start()

    def schedule_rebuild(*_a):
        state["gen"] += 1; state["dirty"] = True; state["last"] = time.time()
        status.set("…")

    def show_images():
        pv_canvas.delete("all"); _pv_photos.clear()
        width = max(pv_canvas.winfo_width(), 320); y = 12
        for _key, img in last_images:
            scale = (width - 24) / img.width
            disp = img if abs(scale - 1) < 0.02 else img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
            photo = ImageTk.PhotoImage(disp); _pv_photos.append(photo)
            pv_canvas.create_image(12, y, anchor="nw", image=photo); y += disp.height + 14
        pv_canvas.configure(scrollregion=(0, 0, width, y))

    def poll():
        try:
            while True:
                g, payload = result_q.get_nowait()
                state["running"] = False
                if isinstance(payload, Exception):
                    status.set(f"error: {payload}")
                elif g == state["gen"]:
                    last_images[:] = payload; show_images(); status.set(f"✓ {len(payload)} page(s)")
        except queue.Empty:
            pass
        if state["dirty"] and not state["running"] and (time.time() - state["last"]) >= 0.12:
            state["dirty"] = False; start_worker(); status.set("rendering…")
        root.after(50, poll)

    _rj = {"id": None}

    def schedule_reshow(*_a):
        if _rj["id"] is not None:
            preview.after_cancel(_rj["id"])
        _rj["id"] = preview.after(120, show_images)

    # ---- var helpers ----
    def set_var(s, item):
        v = tk.BooleanVar(value=item in s)
        v.trace_add("write", lambda *_a: ((s.add if v.get() else s.discard)(item), schedule_rebuild()))
        return v

    def bool_var(d, key):
        v = tk.BooleanVar(value=d[key])
        v.trace_add("write", lambda *_a: (d.__setitem__(key, v.get()), schedule_rebuild()))
        return v

    # =================================================== persistent header (profiles + toolbar)
    ttk.Label(root, text="LIMIT-v2 plot studio", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=8, pady=(8, 2))

    prof = ttk.Frame(root); prof.pack(fill="x", padx=8)
    ttk.Label(prof, text="Profile:").pack(side="left")
    prof_name = tk.StringVar()
    prof_box = ttk.Combobox(prof, textvariable=prof_name, width=18)
    prof_box.pack(side="left", padx=4)

    def list_profiles():
        return sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(PROFILE_DIR, "*.json")))

    def refresh_profiles():
        prof_box["values"] = list_profiles()

    def save_profile():
        name = prof_name.get().strip()
        if not name:
            status.set("name the profile first"); return
        with open(os.path.join(PROFILE_DIR, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(config_to_json(config), f, indent=2)
        refresh_profiles(); status.set(f"saved profile '{name}'")

    def load_profile():
        name = prof_name.get().strip()
        path = os.path.join(PROFILE_DIR, name + ".json")
        if not name or not os.path.exists(path):
            status.set("pick an existing profile"); return
        with open(path, encoding="utf-8") as f:
            new = config_from_json(json.load(f))
        config.clear(); config.update(new); cache.clear()
        build_panel(); schedule_rebuild(); status.set(f"loaded profile '{name}'")

    def delete_profile():
        name = prof_name.get().strip()
        path = os.path.join(PROFILE_DIR, name + ".json")
        if name and os.path.exists(path) and messagebox.askyesno("Delete", f"Delete profile '{name}'?"):
            os.remove(path); refresh_profiles(); prof_name.set(""); status.set(f"deleted '{name}'")

    ttk.Button(prof, text="Save", command=save_profile, style="Accent.TButton").pack(side="left", padx=2)
    ttk.Button(prof, text="Load", command=load_profile).pack(side="left", padx=2)
    ttk.Button(prof, text="Delete", command=delete_profile).pack(side="left", padx=2)

    def export_pdf_dialog():
        path = filedialog.asksaveasfilename(
            title="Export PDF", defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")], initialfile="summary.pdf")
        if not path:
            return
        try:
            export_pdf(copy.deepcopy(config), path)
            status.set(f"exported → {os.path.basename(path)}")
        except Exception as e:
            status.set(f"export failed: {e}")

    tools = ttk.Frame(root); tools.pack(fill="x", padx=8, pady=4)
    ttk.Button(tools, text="Refresh", command=schedule_rebuild).pack(side="left")
    ttk.Button(tools, text="Show preview", command=preview.deiconify).pack(side="left", padx=4)
    ttk.Button(tools, text="Export PDF", command=export_pdf_dialog, style="Accent.TButton").pack(side="left")
    ttk.Label(tools, textvariable=status, style="Hint.TLabel").pack(side="left", padx=8)

    # =================================================== scrollable body
    outer = ttk.Frame(root); outer.pack(fill="both", expand=True, padx=(8, 0))
    cv = tk.Canvas(outer, borderwidth=0, highlightthickness=0, bg=BG)
    sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
    body = ttk.Frame(cv)
    body.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
    cv.create_window((0, 0), window=body, anchor="nw")
    cv.configure(yscrollcommand=sb.set)
    cv.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")

    def _on_wheel(event):
        w = root.winfo_containing(event.x_root, event.y_root)
        target = pv_canvas if (w is not None and w.winfo_toplevel() is preview) else cv
        target.yview_scroll(int(-event.delta / 120), "units")
    root.bind_all("<MouseWheel>", _on_wheel)
    preview.bind("<Configure>", schedule_reshow)

    # ---- per-build UI state ----
    slider_vars = {}
    label_refreshers = []
    guard = {"on": False}

    def add_multiselect(parent, title, items, sel_set, label_fn=str, ncols=4):
        lf = ttk.LabelFrame(parent, text=title); lf.pack(fill="x", padx=6, pady=2)
        bar = ttk.Frame(lf, style="Card.TFrame"); bar.pack(fill="x")
        vmap = {}
        ttk.Button(bar, text="all", width=4, command=lambda: [v.set(True) for v in vmap.values()]).pack(side="left", padx=1, pady=1)
        ttk.Button(bar, text="none", width=5, command=lambda: [v.set(False) for v in vmap.values()]).pack(side="left", padx=1)
        grid = ttk.Frame(lf, style="Card.TFrame"); grid.pack(fill="x")
        for i, it in enumerate(items):
            vmap[it] = set_var(sel_set, it)
            ttk.Checkbutton(grid, text=label_fn(it), variable=vmap[it]).grid(row=i // ncols, column=i % ncols, sticky="w", padx=2)

    def add_parts(parent, c):
        lf = ttk.LabelFrame(parent, text="parts (independent)"); lf.pack(fill="x", padx=6, pady=2)
        ttk.Checkbutton(lf, text="grid (main cells)", variable=bool_var(c, "grid")).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(lf, text="right margin · avg cols", variable=bool_var(c, "mright")).grid(row=0, column=1, sticky="w")
        if "mbottom" in c:
            ttk.Checkbutton(lf, text="bottom margin · avg rows", variable=bool_var(c, "mbottom")).grid(row=1, column=0, sticky="w")
        if "corner" in c:
            ttk.Checkbutton(lf, text="corner · avg both", variable=bool_var(c, "corner")).grid(row=1, column=1, sticky="w")

    # ---- weight sliders ----
    def rebalance(group, moved):
        pos = config["slider_pos"][group]
        others = [k for k in pos if k != moved]
        if not others:
            return
        diff = (1.0 - pos[moved]) - sum(pos[k] for k in others)
        movable = set(others)
        for _ in range(100):
            if abs(diff) < 1e-9 or not movable:
                break
            share = diff / len(movable); leftover = 0.0; still = set()
            for k in list(movable):
                nv = pos[k] + share
                if nv < 0:
                    leftover += nv; pos[k] = 0.0
                elif nv > 1:
                    leftover += nv - 1; pos[k] = 1.0
                else:
                    pos[k] = nv; still.add(k)
            diff = leftover; movable = still
        guard["on"] = True
        for k in others:
            slider_vars[group][k].set(pos[k])
        guard["on"] = False

    def refresh_weight_labels():
        for fn in label_refreshers:
            fn()

    def on_slide(group, k, v):
        if guard["on"]:
            return
        config["slider_pos"][group][k] = float(v)
        if config["normalize"]:
            rebalance(group, k)
        recompute_weights(config); refresh_weight_labels(); schedule_rebuild()

    def on_normalize():
        config["normalize"] = norm_var.get()
        if config["normalize"]:
            for g in ("metrics", "queries", "docs"):
                pos = config["slider_pos"][g]; keys = list(pos)
                s = sum(max(0.0, pos[k]) for k in keys)
                guard["on"] = True
                for k in keys:
                    pos[k] = (max(0.0, pos[k]) / s) if s > 0 else 1.0 / len(keys)
                    slider_vars[g][k].set(pos[k])
                guard["on"] = False
        recompute_weights(config); refresh_weight_labels(); schedule_rebuild()

    def add_weight_group(parent, group, keys, label_fn):
        lf = ttk.LabelFrame(parent, text=group + " weights"); lf.pack(fill="x", padx=6, pady=3)
        slider_vars[group] = {}
        vlabels = {}
        for k in keys:
            row = ttk.Frame(lf, style="Card.TFrame"); row.pack(fill="x", pady=1)
            ttk.Label(row, text=label_fn(k), width=11, style="Card.TLabel").pack(side="left")
            var = tk.DoubleVar(value=config["slider_pos"][group][k]); slider_vars[group][k] = var
            ttk.Scale(row, from_=0, to=1, variable=var, length=150,
                      command=lambda v, k=k, group=group: on_slide(group, k, v)).pack(side="left", padx=4)
            vlabels[k] = ttk.Label(row, text="", width=6, style="Card.TLabel"); vlabels[k].pack(side="left")

        def refresh():
            for k in keys:
                vlabels[k].config(text=f"{config['weights'][group][k]:.3f}")
        label_refreshers.append(refresh); refresh()

    norm_var = tk.BooleanVar(value=config["normalize"])

    # =================================================== panel builder (re-run on profile load)
    def build_panel():
        for ch in body.winfo_children():
            ch.destroy()
        slider_vars.clear(); label_refreshers.clear()
        norm_var.set(config["normalize"])
        P = config["plots"]
        for key in PLOT_ORDER:
            c = P[key]
            frame = ttk.LabelFrame(body, text=PLOT_TITLE[key]); frame.pack(fill="x", padx=4, pady=5)
            ttk.Checkbutton(frame, text="show this plot", variable=bool_var(c, "show")).pack(anchor="w", pady=1)
            if key == "grid_a":
                add_multiselect(frame, "rows · metrics", METRICS, c["rows"], lambda m: ME_LABEL[m])
                add_multiselect(frame, "cols · query types", QTS, c["cols"], lambda q: QT_LABEL[q])
                add_multiselect(frame, "lines · models", MODELS, c["lines"])
                add_multiselect(frame, "x-axis · corpus sizes", NS, c["xvals"], nlab, ncols=6)
                add_parts(frame, c)
            elif key == "grid_b":
                add_multiselect(frame, "rows · models", MODELS, c["rows"])
                add_multiselect(frame, "cols · query types", QTS, c["cols"], lambda q: QT_LABEL[q])
                add_multiselect(frame, "lines · metrics", METRICS, c["lines"], lambda m: ME_LABEL[m])
                add_multiselect(frame, "x-axis · corpus sizes", NS, c["xvals"], nlab, ncols=6)
                add_parts(frame, c)
            elif key == "plot_c":
                bf = ttk.Frame(frame, style="Card.TFrame"); bf.pack(fill="x", padx=6, pady=2)
                ttk.Label(bf, text="band width", style="Card.TLabel").pack(side="left")
                bfv = tk.DoubleVar(value=c["band_factor"])
                bfl = ttk.Label(bf, text=f"{c['band_factor']:.2f}× min/max", width=12, style="Card.TLabel")

                def _bf(v, c=c, bfl=bfl):
                    c["band_factor"] = float(v); bfl.config(text=f"{float(v):.2f}× min/max"); schedule_rebuild()
                ttk.Scale(bf, from_=0, to=1, variable=bfv, length=120, command=_bf).pack(side="left", padx=4)
                bfl.pack(side="left")
                add_multiselect(frame, "bands · query types", QTS, c["bands"], lambda q: QT_LABEL[q])
                add_multiselect(frame, "x-axis · corpus sizes", NS, c["xvals"], nlab, ncols=6)
            elif key == "grid_d":
                add_multiselect(frame, "rows · metrics", METRICS, c["rows"], lambda m: ME_LABEL[m])
                add_multiselect(frame, "cols · corpus sizes", DOC5, c["cols"], nlab)
                add_multiselect(frame, "heatmap y · models", MODELS, c["hm_rows"])
                add_multiselect(frame, "heatmap x · query types", QTS, c["hm_cols"], lambda q: QT_LABEL[q])
                add_parts(frame, c)
            elif key == "pareto":
                add_multiselect(frame, "cols · corpus sizes", DOC5, c["cols"], nlab)
                add_multiselect(frame, "models", MODELS, c["models"])
                add_parts(frame, c)

        wf = ttk.LabelFrame(body, text="averaging weights"); wf.pack(fill="x", padx=4, pady=6)
        ttk.Checkbutton(wf, text="normalize (sliders sum to 1; off → softmax of sliders)",
                        variable=norm_var, command=on_normalize).pack(anchor="w", padx=4, pady=2)
        add_weight_group(wf, "metrics", METRICS, lambda m: ME_LABEL[m])
        add_weight_group(wf, "queries", QTS, lambda q: QT_LABEL[q])
        add_weight_group(wf, "docs", DOC5, nlab)

        add_model_order(body)
        cv.update_idletasks(); cv.configure(scrollregion=cv.bbox("all"))

    # ---- drag-to-reorder model list (applied to every plot) ----
    def add_model_order(parent):
        lf = ttk.LabelFrame(parent, text="model order  ·  drag to reorder"); lf.pack(fill="x", padx=4, pady=6)
        lb = tk.Listbox(lf, height=len(MODELS), activestyle="none", bd=0, highlightthickness=0,
                        selectbackground="#3b6ea5", selectforeground="#fff", font=("Segoe UI", 9))
        lb.pack(fill="x", padx=6, pady=4)
        for m in config["model_order"]:
            lb.insert("end", m)
        drag = {"i": None}

        def press(e):
            drag["i"] = lb.nearest(e.y)

        def motion(e):
            j = lb.nearest(e.y); i = drag["i"]
            if i is None or j == i or not (0 <= j < lb.size()):
                return
            txt = lb.get(i); lb.delete(i); lb.insert(j, txt)
            lb.selection_clear(0, "end"); lb.selection_set(j); drag["i"] = j

        def release(_e):
            config["model_order"] = list(lb.get(0, "end")); schedule_rebuild()

        lb.bind("<Button-1>", press); lb.bind("<B1-Motion>", motion); lb.bind("<ButtonRelease-1>", release)

    refresh_profiles()
    build_panel()
    root.after(50, poll)
    root.mainloop()
