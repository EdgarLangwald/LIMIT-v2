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
from src.generate import generate_dataset
from src.embed import embed_dataset
from src.evaluate import evaluate
from src.plot import visualize_results

# --- dataset ---

N        = 50000        # total distractor documents
M        = 4              # sentences per distractor doc
SEED     = 42

# --- embedding ---

MODELS = [
    "Snowflake_v2",
    "Qwen0.6b",
    ]
BS     = 1000

# --- evaluation ---

N_VALUES = [0, 500, 1000, 2000, 5000] + [10000*i for i in range(1, 6)]# distractor document count <= N
KS       = [1, 5, 10, 50, 200]

# -----------------

dataset, qrels, n_targets, dataset_path = generate_dataset(N, M, seed=SEED)
print(f"{n_targets} targets + {N} distractors = {len(dataset['corpus'])} docs, {len(dataset['queries'])} queries")

for model in MODELS:
    print(f"\n=== {model} ===")

    embs, embs_path = embed_dataset(
        dataset, model, dataset_path,
        batch_size=BS,
        force=False,
    )

    results = evaluate(
        embs, qrels, n_targets, embs_path,
        n_values=N_VALUES,
        ks=KS,
        force=True,
    )

    del embs
    gc.collect()

visualize_results()
