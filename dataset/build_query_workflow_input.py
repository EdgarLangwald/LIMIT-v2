"""Build ground-truth (query, target) pairs for LIMIT-v2 eval benchmarks (v2).

Pipeline (dataset creation):

    1. --dump-specs : deterministically sample everything that does NOT need an LLM
                      (categories, items, target identities, fillers) and write one
                      spec file per *unit* to dataset/query_workflow_input/.
    2. (workflow)   : 100 Haiku agents read one unit spec each and write the queries
                      + template picks to dataset/query_workflow_output/unit_NNN.json
                      (see query_workflow / the Workflow tool). This stage is NOT
                      in this file.
    3. --render     : read specs + agent results, render the final eval dataset.

Structure (the "2 x 2" design):
    - A *unit* = 2 independent draws. There are N units (default 100).
    - A *draw* = K_CATS (3) categories, each with one sampled item.
        * The agent picks EXACTLY TPL_PER_CAT (2) template indices per category.
        * The 2^K_CATS = 8 combinations of those picks -> 8 target profiles.
        * The agent writes 2 queries per draw (specific + broad); both must fit
          BOTH chosen templates of all 3 categories.
    - Each of the 8 targets gets its OWN filler sentence drawn from a distinct
      category not used by the draw.
    => N units -> 2N draw-records, each with 8 targets.

Usage:
    python dataset/build_query_workflow_input.py --dump-specs --seed 42 --n 100
    python dataset/build_query_workflow_input.py --render --seed 42 --out dataset/query_final.json
"""
import argparse
import glob
import json
import os
import random
import re

import numpy as np
from faker import Faker

from src.profile_sentences import generate as _gen, _PRONOUNS
from src.generate import _SAMPLING_CATS, _SAMPLING_WEIGHTS
from src.paths import DATASET_DIR

K_CATS = 3              # categories per draw
TPL_PER_CAT = 2        # template indices the agent picks per category
DRAWS_PER_UNIT = 2     # draws per agent/unit
N_TARGETS = 2 ** K_CATS  # 8 target profiles per draw (all template combinations)


def _eligible_categories():
    """Active categories usable for sampling, with renormalised weights."""
    pairs = [
        (cat, w)
        for cat, w in zip(_SAMPLING_CATS, _SAMPLING_WEIGHTS)
        if cat.pool_size >= 5 and len(cat.templates) >= max(3, TPL_PER_CAT)
    ]
    cats, weights = zip(*pairs)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return list(cats), weights


def _specs_dir(seed: int) -> str:
    return str(DATASET_DIR / "qry_specs")


def _results_dir(seed: int) -> str:
    return str(DATASET_DIR / "qry_raw")


# --------------------------------------------------------------------------- #
# Stage 1: dump specs
# --------------------------------------------------------------------------- #
def dump_specs(n: int, seed: int, out_dir: str) -> None:
    """Deterministically sample n units and write one spec file per unit."""
    cats, elig_w = _eligible_categories()
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    Faker.seed(seed)
    fake = Faker()

    def make_target() -> dict:
        gender = rng.choice(["male", "female"])
        first = rng.choice(
            _gen.male_name.pool if gender == "male" else _gen.female_name.pool
        )
        surname = rng.choice(_gen.family_name.pool)
        dob = fake.date_of_birth(minimum_age=15, maximum_age=90).strftime("%d %B %Y")
        return {
            "name": f"{first} {surname}",
            "gender": gender,
            "dob": dob,
            "state": fake.state(),
            "job": fake.job(),
        }

    os.makedirs(out_dir, exist_ok=True)
    for u in range(n):
        draws = []
        for d in range(DRAWS_PER_UNIT):
            cat_indices = np_rng.choice(len(cats), size=K_CATS, replace=False, p=elig_w)
            chosen = [cats[int(i)] for i in cat_indices]

            slots = [
                {
                    "category_name": cat.category_name,
                    "item": rng.choice(cat.pool),
                    "templates": list(cat.templates),
                }
                for cat in chosen
            ]

            targets_meta = [make_target() for _ in range(N_TARGETS)]

            # One filler per target, each from a DISTINCT category not in the draw.
            chosen_ids = {id(c) for c in chosen}
            filler_candidates = [c for c in cats if id(c) not in chosen_ids]
            filler_cats = rng.sample(filler_candidates, N_TARGETS)
            fillers = [
                {
                    "category_name": fc.category_name,
                    "item": rng.choice(fc.pool),
                    "template": rng.choice(fc.templates),
                    "pos": rng.randint(0, K_CATS),  # insertion index among category sentences
                }
                for fc in filler_cats
            ]

            draws.append({
                "draw_idx": d,
                "slots": slots,
                "targets_meta": targets_meta,
                "fillers": fillers,
            })

        unit = {"unit_idx": u, "draws": draws}
        path = os.path.join(out_dir, f"unit_{u:03d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(unit, f, indent=2, ensure_ascii=False)

    manifest = {
        "seed": seed,
        "n_units": n,
        "draws_per_unit": DRAWS_PER_UNIT,
        "k_cats": K_CATS,
        "tpl_per_cat": TPL_PER_CAT,
        "n_targets": N_TARGETS,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved {n} unit specs -> {out_dir}")


# --------------------------------------------------------------------------- #
# Stage 3: render
# --------------------------------------------------------------------------- #
def _render(template: str, item: str, gender: str) -> str:
    return template.format(item=item, **_PRONOUNS[gender])


def _validate_draw(res_draw: dict, spec_draw: dict, where: str) -> None:
    """Hard-validate one agent draw result against its spec. Raises ValueError."""
    cats = res_draw.get("categories")
    if not isinstance(cats, list) or len(cats) != K_CATS:
        raise ValueError(f"{where}: expected {K_CATS} categories, got {cats!r}")
    for c_idx, (cat_res, slot) in enumerate(zip(cats, spec_draw["slots"])):
        if cat_res.get("category_name") != slot["category_name"]:
            raise ValueError(
                f"{where} cat{c_idx}: name mismatch "
                f"{cat_res.get('category_name')!r} != {slot['category_name']!r}"
            )
        idxs = cat_res.get("template_indices")
        n_tmpl = len(slot["templates"])
        if not isinstance(idxs, list) or len(idxs) != TPL_PER_CAT:
            raise ValueError(f"{where} cat{c_idx}: need {TPL_PER_CAT} indices, got {idxs!r}")
        for idx in idxs:
            if not isinstance(idx, int) or not (0 <= idx < n_tmpl):
                raise ValueError(
                    f"{where} cat{c_idx}: index {idx!r} out of range [0,{n_tmpl - 1}]"
                )
        if len(set(idxs)) != TPL_PER_CAT:
            raise ValueError(f"{where} cat{c_idx}: indices must be distinct, got {idxs}")
    for key in ("query_specific", "query_broad"):
        q = res_draw.get(key)
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"{where}: missing/empty {key}")

    # No query may name the exact item ("describe, don't name"). Word-boundary,
    # case-insensitive. Note: over-generic broad queries are a semantic failure
    # the generator prompt must prevent — they cannot be auto-detected here.
    items = [slot["item"] for slot in spec_draw["slots"]]
    for key in ("query_specific", "query_broad"):
        q = res_draw[key]
        leaked = [it for it in items if re.search(rf"\b{re.escape(it)}\b", q, re.I)]
        if leaked:
            raise ValueError(f"{where}: {key} names item(s) {leaked}")


def _build_targets(spec_draw: dict, res_draw: dict) -> list[dict]:
    """Render the 8 target profiles for one draw (all template combinations)."""
    slots = spec_draw["slots"]
    picks = [res_draw["categories"][c]["template_indices"] for c in range(K_CATS)]

    targets = []
    for t in range(N_TARGETS):
        meta = spec_draw["targets_meta"][t]
        gender = meta["gender"]
        intro = (
            f"{meta['name']} was born on {meta['dob']},"
            f" lives in {meta['state']}, and works as a {meta['job']}."
        )

        cat_sentences, cat_facts = [], []
        for c in range(K_CATS):
            bit = (t >> c) & 1                      # which of the 2 chosen templates
            tmpl_idx = picks[c][bit]
            slot = slots[c]
            cat_sentences.append(_render(slot["templates"][tmpl_idx], slot["item"], gender))
            cat_facts.append({"category": slot["category_name"], "item": slot["item"]})

        filler = spec_draw["fillers"][t]
        filler_sentence = _render(filler["template"], filler["item"], gender)
        filler_fact = {"category": filler["category_name"], "item": filler["item"]}

        pos = filler["pos"]
        sentences = cat_sentences[:pos] + [filler_sentence] + cat_sentences[pos:]
        facts = cat_facts[:pos] + [filler_fact] + cat_facts[pos:]

        targets.append({
            "name": meta["name"],
            "gender": gender,
            "sentences": [intro] + sentences,
            "facts": [
                {
                    "category": "intro",
                    "name": meta["name"],
                    "dob": meta["dob"],
                    "state": meta["state"],
                    "job": meta["job"],
                }
            ] + facts,
        })
    return targets


def render_from_specs(seed: int, out_path: str) -> None:
    """Render the final eval dataset from spec files + agent result files."""
    spec_dir, res_dir = _specs_dir(seed), _results_dir(seed)
    spec_files = sorted(glob.glob(os.path.join(spec_dir, "unit_*.json")))
    if not spec_files:
        raise FileNotFoundError(f"No spec files in {spec_dir} (run --dump-specs first)")

    output, skipped = [], []
    for spec_path in spec_files:
        with open(spec_path, encoding="utf-8-sig") as f:
            spec = json.load(f)
        u = spec["unit_idx"]

        res_path = os.path.join(res_dir, f"unit_{u:03d}.json")
        if not os.path.exists(res_path):
            skipped.append(f"unit {u}: missing result file")
            continue
        try:
            # utf-8-sig: some agents write the file with a UTF-8 BOM.
            with open(res_path, encoding="utf-8-sig") as f:
                res = json.load(f)
        except json.JSONDecodeError as e:
            skipped.append(f"unit {u}: bad JSON ({e})")
            continue

        res_by_draw = {dr.get("draw_idx"): dr for dr in res.get("draws", [])}
        for spec_draw in spec["draws"]:
            d = spec_draw["draw_idx"]
            res_draw = res_by_draw.get(d)
            where = f"unit {u} draw {d}"
            if res_draw is None:
                skipped.append(f"{where}: missing in result")
                continue
            try:
                _validate_draw(res_draw, spec_draw, where)
            except ValueError as e:
                skipped.append(str(e))
                continue

            slots_out = [
                {
                    "category_name": slot["category_name"],
                    "item": slot["item"],
                    "template_indices": res_draw["categories"][c]["template_indices"],
                }
                for c, slot in enumerate(spec_draw["slots"])
            ]
            output.append({
                "draw_id": f"{u}-{d}",
                "unit_idx": u,
                "draw_idx": d,
                "query_specific": res_draw["query_specific"].strip(),
                "query_broad": res_draw["query_broad"].strip(),
                "slots": slots_out,
                "targets": _build_targets(spec_draw, res_draw),
            })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(output)} draw-records ({N_TARGETS} targets each) -> {out_path}")
    if skipped:
        print(f"\nSkipped {len(skipped)} draw(s) — re-run those units:")
        for s in skipped:
            print(f"  - {s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=100, help="number of units (dump-specs)")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--dump-specs", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.dump_specs:
        dump_specs(args.n, args.seed, args.out or _specs_dir(args.seed))
        return
    if args.render:
        out_path = args.out or str(DATASET_DIR / "qry_final.json")
        render_from_specs(args.seed, out_path)
        return
    parser.error("choose one of --dump-specs or --render")


if __name__ == "__main__":
    main()
