"""LIMIT-v2 pipeline: generate dataset → embed with different models
   → evaluate at increasing distractor counts → visualize"""
import os

# load huggingface token to download models from hf with authentification
def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

import gc
import json
from src.generate import generate_dataset
from src.distort_dataset import add_random_chars
from src.embed import embed_dataset, load_model
from src.evaluate import evaluate
from src.plot import visualize_results
from src.paths import RESULTS_DIR

# --- dataset ---

N        = 500000           # total distractor documents
M        = 4              # sentences per distractor doc
SEED     = 42

original, qrels, n_targets, query_types, dataset_path = generate_dataset(N, M, seed=SEED)
den1_spr02_rep0 = add_random_chars(original, density=0.01, replace=False, density_spread=0.002, seed=SEED)
den1_spr02_rep1 = add_random_chars(original, density=0.01, replace=True, density_spread=0.002, seed=SEED)
den2_spr04_rep0 = add_random_chars(original, density=0.02, replace=False, density_spread=0.004, seed=SEED)
den2_spr04_rep1 = add_random_chars(original, density=0.02, replace=True, density_spread=0.004, seed=SEED)
den4_spr08_rep0 = add_random_chars(original, density=0.04, replace=False, density_spread=0.008, seed=SEED)
den4_spr08_rep1 = add_random_chars(original, density=0.04, replace=True, density_spread=0.008, seed=SEED)
print(f"{n_targets} targets + {N} distractors = {len(original['corpus'])} docs, {len(original['queries'])} queries")

DATASETS = {
    "original":        original,
    "den1_spr02_rep0": den1_spr02_rep0,
    "den1_spr02_rep1": den1_spr02_rep1,
    "den2_spr04_rep0": den2_spr04_rep0,
    "den2_spr04_rep1": den2_spr04_rep1,
    "den4_spr08_rep0": den4_spr08_rep0,
    "den4_spr08_rep1": den4_spr08_rep1,
}

# --- embedding ---

MODELS = [
    "Snowflake_v2",     # 568M
    "Qwen0.6b",         # 596M
    "Qwen8b",           # 7.6B
    "Promptriever",     # 8.0B
    "Jina_v5",          # 677M (Qwen3-0.6B-Base)
    "GritLM",           # 7.2B (Mistral-7B)
    "Octen4b",
]
BS = 100

# --- evaluation ---

N_VALUES = [0, 300, 1000, 3000, 10000, 30000] + [100000*i for i in range(1, 6)]
KS       = [1, 5, 10, 50, 200]

type_order = list(dict.fromkeys(query_types))
OUT_DIR    = RESULTS_DIR / "add_random_chars"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _save_partial(model_name, collected, processed):
    payload = {
        "name":                   model_name,
        "ks":                     KS,
        "n_values":               sorted(collected) if collected else [],
        "query_types":            type_order,
        "distortions":            list(DATASETS.keys()),
        "processed_distortions":  sorted(processed),
        "results":                {str(n): collected[n] for n in sorted(collected)},
    }
    with open(OUT_DIR / f"{model_name}_partial.json", "w") as f:
        json.dump(payload, f, indent=2)

for model_name in MODELS:
    final_path   = OUT_DIR / f"{model_name}.json"
    partial_path = OUT_DIR / f"{model_name}_partial.json"

    if final_path.exists():
        print(f"\n=== {model_name} (already done, skipping) ===")
        continue

    # Resume from partial checkpoint if it exists
    if partial_path.exists():
        with open(partial_path) as f:
            saved = json.load(f)
        processed = set(saved.get("processed_distortions", []))
        collected = {int(n): data for n, data in saved["results"].items()}
        print(f"\n=== {model_name} (resuming; done: {sorted(processed)}) ===")
    else:
        processed = set()
        collected = {}
        print(f"\n=== {model_name} ===")

    loaded_model  = load_model(model_name)
    orig_qry_embs = None

    for dist_name, dataset in DATASETS.items():
        if dist_name in processed:
            print(f"  -- {dist_name} (skipped)")
            continue

        print(f"  -- {dist_name}")
        if dist_name == "original":
            ds_path    = dataset_path
            only_embed = None
        else:
            ds_path    = dataset_path.parent / f"{dataset_path.stem}__{dist_name}.json"
            only_embed = "docs"

        embs, _ = embed_dataset(
            dataset, model_name, ds_path,
            batch_size=BS,
            force=False,
            model=loaded_model,
            only_embed=only_embed,
        )

        if dist_name == "original":
            orig_qry_embs = embs["qry_embs"]
        else:
            if orig_qry_embs is None:
                # original was processed in a previous run; recover query embs from cache
                orig_embs_tmp, _ = embed_dataset(
                    original, model_name, dataset_path,
                    force=False, model=loaded_model,
                )
                orig_qry_embs = orig_embs_tmp["qry_embs"]
            embs = {"doc_embs": embs["doc_embs"], "qry_embs": orig_qry_embs}

        dist_results = evaluate(
            embs, qrels, n_targets,
            embs_path=None,
            query_types=query_types,
            n_values=N_VALUES,
            ks=KS,
            force=True,
        )

        for n, metrics in dist_results.items():
            if n not in collected:
                collected[n] = {}
            for metric, vals in metrics.items():
                if metric not in collected[n]:
                    collected[n][metric] = {qt: {} for qt in type_order}
                for qt, val in zip(type_order, vals):
                    collected[n][metric][qt][dist_name] = val

        del embs
        gc.collect()

        processed.add(dist_name)
        _save_partial(model_name, collected, processed)

    del loaded_model
    gc.collect()

    # Write final JSON and remove partial checkpoint
    final_payload = {
        "name":        model_name,
        "ks":          KS,
        "n_values":    sorted(collected),
        "query_types": type_order,
        "distortions": list(DATASETS.keys()),
        "results":     {str(n): collected[n] for n in sorted(collected)},
    }
    with open(final_path, "w") as f:
        json.dump(final_payload, f, indent=2)
    partial_path.unlink(missing_ok=True)
    print(f"  Saved results/add_random_chars/{model_name}.json")
