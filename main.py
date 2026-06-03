"""LIMIT-v2 full pipeline: generate dataset → embed → interactive evaluation."""
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

from src.generate import generate_dataset
from src.embed import embed_dataset, load_model
from src.evaluate import evaluate_manually

# --- configure here before running ---
MODEL  = "Snowflake_v2"
N      = 10000
M      = 4
SEED   = 42
K      = 10
FORCE  = False
# -------------------------------------

dataset_name = f"n{N}_m{M}_s{SEED}"

# 1. Generate
print(f"\n=== Generate ===")
docs = generate_dataset(N, M, seed=SEED)
print(f"{len(docs)} documents ready.")

out_dir = os.path.join(os.path.dirname(__file__), "dataset")
os.makedirs(out_dir, exist_ok=True)
txt_path = os.path.join(out_dir, f"{dataset_name}.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join("\n".join(doc["sentences"]) for doc in docs))
print(f"Saved to {txt_path}\n")

# 2. Embed
print(f"=== Embed ===")
doc_embs = embed_dataset(docs, MODEL, dataset_name, force=FORCE)
print(f"Embeddings: {doc_embs.shape}\n")

# 3. Load model for query embedding
print(f"=== Load model for queries ===")
model = load_model(MODEL)
print()

# 4. Interactive eval loop
print(f"=== Evaluate ===")
print(f"Top-{K} results, {len(docs)} docs total. Type 'quit' to exit.\n")

while True:
    sentence = input("Sentence : ").strip()
    if sentence.lower() in ("quit", "q", "exit", ""):
        break
    target = input("Target   : ").strip()
    if not target:
        continue

    result = evaluate_manually(sentence, target, doc_embs, docs, MODEL, k=K, model=model)

    print(f"\nTop {K}:")
    for rank, name in result["top_k"]:
        marker = "  <-- target" if name.lower() == target.lower() else ""
        print(f"  {rank:>3}. {name}{marker}")

    if result["target_rank"] is not None:
        print(f"\n  Target rank: {result['target_rank']} / {len(docs)}")
    else:
        print(f"\n  '{target}' not found in dataset (check spelling)")
    print()
