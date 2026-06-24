"""Reproduce RTEB (HuggingFace Retrieval Embedding Benchmark) scores to validate that our
model-loading / embedding pipeline is configured correctly.

Strategy: reuse the SAME code under test — ``src.embed.load_model`` + ``src.embed.embed_dataset``
— to embed an open RTEB dataset, score it with ``src.evaluate.evaluate_retrieval`` (NDCG@10 via
pytrec_eval, the BEIR/MTEB/RTEB engine), and diff against the numbers RTEB published. A close match
proves the model is downloaded, prompted, truncated and normalized the way RTEB does it; a gap
localizes a config issue.

Dataset: AILAStatutes — RTEB's smallest tier-0 (fully open) dataset, 82 docs / 50 queries.

Config note: we deliberately DO NOT edit ``embed.py``'s MODELS dict (so the synthetic LIMIT runs in
main.py keep their person-profile prompts). Instead the RTEB-faithful prompts are injected into the
live MODELS dict in-process (RTEB_PROMPTS), and RTEB's truncation caps are applied by pre-loading
each model and setting ``max_seq_length`` before handing it to embed_dataset. The only persistent
embed.py edit is the Jina v3->v5 swap.

Usage:
    PYTHONPATH=. python reproduce.py                 # all models
    PYTHONPATH=. python reproduce.py --models BGE_M3,Qwen8b
    PYTHONPATH=. python reproduce.py --force         # ignore embedding cache
"""
import argparse
import gc
import json
import os
import urllib.request
from pathlib import Path


def _load_env():
    """Load HF token from .env so gated model downloads authenticate (mirrors main.py)."""
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

import numpy as np

from src.embed import MODELS, load_model, embed_dataset
from src.evaluate import evaluate_retrieval
from src.paths import DATASET_DIR, RESULTS_DIR

# --- benchmark target ---------------------------------------------------------------------------

RTEB_DATASET = "AILAStatutes"                                  # smallest tier-0 open RTEB dataset
HF_DATASET_REPO = f"embedding-benchmark/{RTEB_DATASET}"
RESULTS_JSON_URL = "https://raw.githubusercontent.com/embedding-benchmark/rteb/main/results/results.json"
# Minimal offline copy of the published targets: {our_model_name: ndcg_at_10} for models that have
# one. Generated from RESULTS_JSON_URL by `--refresh-targets`; read at compare time so the run needs
# no network for scoring. (Lives under the gitignored dataset/rteb/ with the downloaded jsonl.)
PUBLISHED_PATH = DATASET_DIR / "rteb" / f"published_{RTEB_DATASET}.json"

# Models to run. All are embedded + scored; those without a committed per-dataset target just show
# "—" in the published column (they aren't on RTEB, or only have a live-leaderboard aggregate).
SELECTION = [
    "Snowflake_v2", "Qwen0.6b", "Qwen8b", "BGE_M3", "GritLM",
    "Octen_0.6b", "Octen_4b", "Jina_v5", "Voyage_4_nano", "Nomic_Embed_v2", "Promptriever",
]

# --- RTEB-faithful config (in-process overrides; embed.py on disk is untouched) ------------------

# Prompts exactly as RTEB applies them (read from rteb/models/*.py). The surprise: RTEB sets NO
# query instruction for Qwen3 / Octen, and none for GritLM. Snowflake/Qwen-0.6B aren't in RTEB's
# committed repo -> use each model's documented retrieval prompt. Nomic_Embed_v2 / Promptriever are
# NOT on RTEB at all -> left out of this map so they keep their existing embed.py config as-is.
RTEB_PROMPTS = {
    "BGE_M3":       {"hf_id": "BAAI/bge-m3", "query_prefix": ""},
    "Qwen0.6b":     {"hf_id": "Qwen/Qwen3-Embedding-0.6B", "query_prefix": ""},
    "Qwen8b":       {"hf_id": "Qwen/Qwen3-Embedding-8B", "query_prefix": ""},
    "Snowflake_v2": {"hf_id": "Snowflake/snowflake-arctic-embed-l-v2.0", "query_prefix": "query: "},
    "GritLM":       {"hf_id": "GritLM/GritLM-7B", "query_prefix": "<|embed|>\n",
                     "doc_prefix": "<|embed|>\n", "gritlm_api": True},  # drop the <|user|> instruction
    "Octen_0.6b":   {"hf_id": "Octen/Octen-Embedding-0.6B", "query_prefix": ""},
    "Octen_4b":     {"hf_id": "Octen/Octen-Embedding-4B", "query_prefix": ""},
    # Jina_v5 lives in embed.py (persistent swap); no override needed.
}

# Max input length (tokens) RTEB caps each model to (model_meta.max_tokens). Applied to the loaded
# model so long statutes aren't silently truncated at the SentenceTransformer default. GritLM is
# omitted (it routes through _GritLMEmbedder, which fixes its own max_length).
RTEB_MAXLEN = {
    "BGE_M3": 8192, "Qwen0.6b": 32768, "Qwen8b": 32000, "Snowflake_v2": 8192,
    "Octen_0.6b": 32768, "Octen_4b": 32768, "Jina_v5": 32768, "Voyage_4_nano": 32000,
}

# Our model name -> the model_name used in RTEB's results.json (for the published-score lookup).
RTEB_NAME = {
    "BGE_M3": "bge-m3",
    "Qwen0.6b": "Qwen3-Embedding-0.6B",
    "Qwen8b": "Qwen3-Embedding-8B",
    "Snowflake_v2": "snowflake-arctic-embed-l-v2.0",
    "GritLM": "GritLM-7B",
    "Octen_0.6b": "Octen-Embedding-0.6B",
    "Octen_4b": "Octen-Embedding-4B",
    "Jina_v5": "jina-embeddings-v5-text-small",
    "Voyage_4_nano": "voyage-4-nano",
    "Nomic_Embed_v2": "nomic-embed-text-v2-moe",
    "Promptriever": "promptriever-llama3.1-8b-instruct-v1",
}


# --- data ----------------------------------------------------------------------------------------

def _download_dataset(dest: Path) -> Path:
    """Ensure corpus/queries/relevance jsonl are present under ``dest``. Returns the dataset dir."""
    files = ["corpus.jsonl", "queries.jsonl", "relevance.jsonl"]
    dest.mkdir(parents=True, exist_ok=True)
    if all((dest / f).exists() for f in files):
        return dest
    try:
        from huggingface_hub import hf_hub_download
        for f in files:
            hf_hub_download(repo_id=HF_DATASET_REPO, filename=f, repo_type="dataset",
                            local_dir=str(dest))
    except ImportError:
        # huggingface_hub absent (e.g. a metric-only venv) — the files are public, fetch directly.
        base = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main"
        for f in files:
            print(f"  downloading {f} ...")
            urllib.request.urlretrieve(f"{base}/{f}", dest / f)
    return dest


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rteb_dataset(data_dir: Path):
    """Return (dataset, qrels, doc_ids, query_ids) in the shape embed_dataset/evaluate_retrieval use.

    dataset:   {"corpus": {doc_id: text}, "queries": {query_id: text}}
    qrels:     {query_id: {doc_id: relevance_int}}
    """
    corpus_rows = _read_jsonl(data_dir / "corpus.jsonl")
    query_rows  = _read_jsonl(data_dir / "queries.jsonl")
    rel_rows    = _read_jsonl(data_dir / "relevance.jsonl")

    def _doc_text(r: dict) -> str:
        title, text = r.get("title", "") or "", r.get("text", "") or ""
        return f"{title}\n{text}".strip() if title else text

    corpus  = {r["id"]: _doc_text(r)   for r in corpus_rows}
    queries = {r["id"]: r["text"]      for r in query_rows}

    qrels: dict[str, dict[str, int]] = {}
    for r in rel_rows:
        qid, cid, score = r["query-id"], r["corpus-id"], int(r.get("score", 1))
        qrels.setdefault(qid, {})[cid] = score

    dataset = {"corpus": corpus, "queries": queries}
    return dataset, qrels, list(corpus.keys()), list(queries.keys())


def published_scores() -> dict[str, float]:
    """Read the minimal local targets file -> {our_model_name: ndcg_at_10}. No network.

    Returns {} (with a hint) if the file is absent — run `--refresh-targets` once while online.
    """
    if not PUBLISHED_PATH.exists():
        print(f"  (no local targets at {PUBLISHED_PATH}; run with --refresh-targets while online)")
        return {}
    with open(PUBLISHED_PATH, encoding="utf-8") as f:
        return json.load(f).get("scores", {})


def refresh_targets() -> dict[str, float]:
    """Fetch RTEB's results.json, distil it to {our_model_name: ndcg_at_10} for models that have a
    score, and write the minimal local file. Requires network."""
    with urllib.request.urlopen(RESULTS_JSON_URL, timeout=30) as resp:
        data = json.load(resp)
    entry = next((d for d in data if d.get("dataset_name") == RTEB_DATASET), None)
    by_rteb = {r["model_name"]: r.get("ndcg_at_10") for r in (entry or {}).get("results", [])
               if r.get("ndcg_at_10") is not None}
    scores = {ours: round(by_rteb[rteb], 5) for ours, rteb in RTEB_NAME.items() if rteb in by_rteb}
    PUBLISHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PUBLISHED_PATH, "w", encoding="utf-8") as f:
        json.dump({"dataset": RTEB_DATASET, "metric": "ndcg_at_10",
                   "source": RESULTS_JSON_URL, "scores": scores}, f, indent=2)
    print(f"  refreshed {PUBLISHED_PATH}: {len(scores)} targets")
    return scores


# --- run -----------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated subset of model names (default: all in SELECTION)")
    ap.add_argument("--topk", type=int, default=100, help="docs retrieved per query before scoring")
    ap.add_argument("--force", action="store_true", help="ignore the embedding cache and re-embed")
    ap.add_argument("--refresh-targets", action="store_true",
                    help="re-download the published targets file (needs network), then continue")
    args = ap.parse_args()

    if args.refresh_targets:
        refresh_targets()

    models = [m.strip() for m in args.models.split(",")] if args.models else list(SELECTION)

    # Inject RTEB-faithful prompts into the live registry (in-process only; embed.py is untouched).
    MODELS.update(RTEB_PROMPTS)

    data_dir = _download_dataset(DATASET_DIR / "rteb" / RTEB_DATASET)
    dataset, qrels, doc_ids, query_ids = load_rteb_dataset(data_dir)
    print(f"{RTEB_DATASET}: {len(doc_ids)} docs, {len(query_ids)} queries, "
          f"{sum(len(v) for v in qrels.values())} relevance judgments")

    # Cache key for embeddings: dataset stem "AILAStatutes" -> embeddings/AILAStatutes/<model>.
    dataset_path = DATASET_DIR / "rteb" / f"{RTEB_DATASET}.json"

    published = published_scores()
    rows = []
    for name in models:
        if name not in MODELS:
            print(f"\n=== {name} ===  (unknown model name, skipping)")
            continue
        print(f"\n=== {name} ===")
        model = load_model(name)
        if name in RTEB_MAXLEN and hasattr(model, "max_seq_length"):
            print(f"  max_seq_length: {model.max_seq_length} -> {RTEB_MAXLEN[name]}")
            model.max_seq_length = RTEB_MAXLEN[name]

        embs, _ = embed_dataset(dataset, name, dataset_path, model=model, force=args.force)
        scores = evaluate_retrieval(embs, qrels, doc_ids, query_ids, ks=(10,), top_k=args.topk,
                                    results_dir=RESULTS_DIR / "rteb",
                                    name=f"{RTEB_DATASET}_{name}")
        computed = scores[10]["ndcg@10"]
        ref = published.get(name)                       # local targets are keyed by our model names
        rows.append((name, computed, ref))
        print(f"  NDCG@10={computed:.5f}  recall@10={scores[10]['recall@10']:.5f}  "
              f"map@10={scores[10]['map@10']:.5f}")

        del embs, model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # --- comparison table ---
    print(f"\n{'='*64}\nRTEB {RTEB_DATASET} — NDCG@10: computed vs published\n{'='*64}")
    print(f"{'model':<16}{'computed':>12}{'published':>12}{'Δ':>12}")
    print("-" * 52)
    for name, computed, ref in rows:
        if ref is None:
            print(f"{name:<16}{computed:>12.5f}{'—':>12}{'—':>12}")
        else:
            print(f"{name:<16}{computed:>12.5f}{ref:>12.5f}{computed - ref:>+12.5f}")
    print("-" * 52)
    print("'—' = no committed per-dataset target in RTEB results.json (check the live HF leaderboard).")


if __name__ == "__main__":
    main()
