"""LIMIT-v2 pipeline: generate dataset → embed → evaluate at increasing distractor counts."""
import os

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
from src.generate import generate_dataset
from src.embed import embed_dataset
from src.evaluate import evaluate, plot_results, plot_mrr_comparison

# --- dataset ---

N        = 100000        # total distractor documents
M        = 4              # sentences per distractor doc
SEED     = 42

# --- embedding ---

MODELS = [
    "Snowflake_v2",
    "Qwen0.6b",
    ]
BS     = 512
DEVICE = "cuda"

# --- evaluation ---

N_VALUES = [0, 500, 1000, 2000, 5000] + [10000*i for i in range(1, 11)]# distractor document count <= N
KS       = [1, 5, 10, 50, 200]
FORCE    = False

# -----------------

dataset_name = f"n{N}_m{M}_s{SEED}"

dataset, qrels, n_targets = generate_dataset(N, M, seed=SEED)
print(f"{n_targets} targets + {N} distractors = {len(dataset['corpus'])} docs, {len(dataset['queries'])} queries")

all_results = {}
for model in MODELS:
    print(f"\n=== {model} ===")
    embs     = embed_dataset(dataset, model, dataset_name, force=FORCE, batch_size=BS, device=DEVICE)
    doc_embs = embs["doc_embs"]
    qry_embs = embs["qry_embs"]

    results = evaluate(doc_embs, qry_embs, qrels, n_targets, N_VALUES, ks=KS)
    all_results[model] = results

    for n, metrics in sorted(results.items()):
        recalls = "  ".join(f"R@{k}={metrics[f'recall@{k}']:.3f}" for k in KS)
        print(f"  n={n:>6}  MRR={metrics['mrr']:.3f}  {recalls}")

    plot_results(results, title=f"{model} — {dataset_name}", show=False)
    del doc_embs, qry_embs, embs
    gc.collect()

plot_mrr_comparison(all_results, title=f"MRR — {dataset_name}", show=True)
