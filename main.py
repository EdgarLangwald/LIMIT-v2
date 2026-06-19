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

N        = 5000        # total distractor documents
M        = 4              # sentences per distractor doc
SEED     = 42

# --- embedding ---

MODELS = [
    "Nomic_Embed_v2",   # 475M (305M active, MoE)
    "Snowflake_v2",     # 568M
    "Qwen0.6b",         # 596M
    "Qwen8b",           # 7.6B
    "Promptriever",     # 8.0B
    "BGE_M3",           # 569M
    "Jina_v3",          # 572M
    "GritLM",           # 7.2B (Mistral-7B)
    ]
BS     = 100

# --- evaluation ---

N_VALUES = [0, 500, 1000, 2000, 5000] #+ [10000*i for i in range(1, 6)] # distractor document count <= N
KS       = [1, 5, 10, 50, 200]

# -----------------

dataset, qrels, n_targets, query_types, dataset_path = generate_dataset(N, M, seed=SEED)
print(f"{n_targets} targets + {N} distractors = {len(dataset['corpus'])} docs, {len(dataset['queries'])} queries")

for model in MODELS:
    print(f"\n=== {model} ===")

    embs, embs_path = embed_dataset(
        dataset, model, dataset_path,
        batch_size=BS,
        force=True,
    )

    
    results = evaluate(
        embs, qrels, n_targets, embs_path, query_types,
        n_values=N_VALUES,
        ks=KS,
        force=True,
    )
    
    
    del embs
    gc.collect()

visualize_results()
