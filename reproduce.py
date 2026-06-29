"""Validate embedding pipeline against MTEB NDCG@10 scores.

Dataset  : mteb/AILA_statutes — 82 docs, 50 queries, 217 qrels.
Reference: MTEB(Law, v1) leaderboard CSV, NDCG@10, queried 2026-06-25.

Note: uses evaluate_retrieval() (not evaluate()) because MTEB qrels are
{query_id: {doc_id: score}} dicts with variable relevance counts per query,
which is incompatible with evaluate()'s fixed-count positional index format.
"""
import gc, json, os, urllib.parse, urllib.request
from pathlib import Path
import numpy as np

from src.embed import MODELS, embed_dataset
from src.evaluate import evaluate_retrieval
from src.paths import DATASET_DIR, RESULTS_DIR

_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

SELECTION = [
    "Snowflake_v2", "Qwen0.6b", "Qwen8b", "BGE_M3", "GritLM",
    "Octen_0.6b", "Octen_4b", "Jina_v5", "Voyage_4_nano", "Nomic_Embed_v2", "Promptriever",
]

PUBLISHED = {   # NDCG@10 from MTEB(Law, v1) leaderboard CSV, queried 2026-06-25
    "BGE_M3": 0.2904, "Qwen8b": 0.8509, "GritLM": 0.4180, "Octen_4b": 0.8557,
    "Jina_v5": 0.5330, "Snowflake_v2": 0.2282, "Qwen0.6b": 0.7902, "Octen_0.6b": 0.8659,
}

# Override embed.py's person-profile prompts with MTEB's evaluation config.
# Octen_0.6b / Octen_4b are not in embed.py — added here as full entries.
_LEGAL_INSTRUCT = "Given a legal scenario, retrieve the most relevant statute documents"
MTEB_PROMPTS = {
    "BGE_M3":       {"hf_id": "BAAI/bge-m3",                             "query_prefix": ""},
    "Qwen0.6b":     {"hf_id": "Qwen/Qwen3-Embedding-0.6B",
                     "query_prefix": f"Instruct: {_LEGAL_INSTRUCT}\nQuery: "},
    "Qwen8b":       {"hf_id": "Qwen/Qwen3-Embedding-8B",
                     "query_prefix": f"Instruct: {_LEGAL_INSTRUCT}\nQuery: "},
    "Snowflake_v2": {"hf_id": "Snowflake/snowflake-arctic-embed-l-v2.0", "query_prefix": "query: "},
    "GritLM":       {"hf_id": "GritLM/GritLM-7B",
                     "query_prefix": "<|embed|>\n", "doc_prefix": "<|embed|>\n", "gritlm_api": True},
    "Octen_0.6b":   {"hf_id": "Octen/Octen-Embedding-0.6B",
                     "query_prefix": f"Instruct: {_LEGAL_INSTRUCT}\nQuery:",
                     "doc_prefix": " "},
    "Octen_4b":     {"hf_id": "Octen/Octen-Embedding-4B",
                     "query_prefix": f"Instruct: {_LEGAL_INSTRUCT}\nQuery:",
                     "doc_prefix": " "},
    "Promptriever": {"hf_id": "samaya-ai/promptriever-llama3.1-8b-instruct-v1",
                     "query_prefix": "query:  ",
                     "query_suffix": f" {_LEGAL_INSTRUCT}.",
                     "doc_prefix": "passage:  "},
}


def _fetch_rows(config, split):
    enc = urllib.parse.quote("mteb/AILA_statutes")
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        d = json.loads(urllib.request.urlopen(
            f"https://datasets-server.huggingface.co/rows?dataset={enc}"
            f"&config={config}&split={split}&offset={offset}&length=100"
        ).read())
        total   = d["num_rows_total"]
        rows   += [r["row"] for r in d["rows"]]
        offset += len(d["rows"])
    return rows


def load_dataset():
    """Fetch mteb/AILA_statutes if not cached, save combined JSON, return for evaluate_retrieval."""
    d = DATASET_DIR / "mteb" / "AILA_statutes"
    d.mkdir(parents=True, exist_ok=True)
    for fname, cfg, spl in [("corpus.jsonl",    "corpus",  "corpus"),
                              ("queries.jsonl",   "queries", "queries"),
                              ("relevance.jsonl", "default", "test")]:
        if not (d / fname).exists():
            rows = _fetch_rows(cfg, spl)
            with open(d / fname, "w") as f:
                for row in rows: f.write(json.dumps(row) + "\n")

    def read(fname):
        return [json.loads(l) for l in open(d / fname) if l.strip()]

    corpus  = {r["_id"]: ((r.get("title") or "") + "\n" + (r.get("text") or "")).strip()
               for r in read("corpus.jsonl")}
    queries = {r["_id"]: r["text"] for r in read("queries.jsonl")}
    qrels   = {}
    for r in read("relevance.jsonl"):
        qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r.get("score", 1))

    dataset      = {"corpus": corpus, "queries": queries}
    dataset_path = d.parent / "AILA_statutes.json"
    dataset_path.write_text(json.dumps(dataset, indent=2))

    return dataset, qrels, list(corpus), list(queries), dataset_path


def main():
    MODELS_TO_RUN = list(SELECTION)
    BS    = 64
    FORCE = False

    MODELS.update(MTEB_PROMPTS)
    dataset, qrels, doc_ids, query_ids, dataset_path = load_dataset()
    print(f"AILA_statutes: {len(doc_ids)} docs, {len(query_ids)} queries, "
          f"{sum(len(v) for v in qrels.values())} relevance judgments")

    rows = []
    for model in MODELS_TO_RUN:
        print(f"\n=== {model} ===")

        embs, _ = embed_dataset(dataset, model, dataset_path, batch_size=BS, force=FORCE)

        scores = evaluate_retrieval(
            embs, qrels, doc_ids, query_ids,
            ks=(10,), top_k=100,
            results_dir=RESULTS_DIR / "mteb",
            name=f"AILA_statutes_{model}",
        )

        computed = scores[10]["ndcg@10"]
        rows.append((model, computed))
        print(f"  NDCG@10={computed:.5f}  recall@10={scores[10]['recall@10']:.5f}  "
              f"map@10={scores[10]['map@10']:.5f}")
        del embs
        gc.collect()

    print(f"\n{'='*52}\nMTEB AILA_statutes — NDCG@10\n{'='*52}")
    print(f"{'model':<16}{'computed':>10}{'published':>11}{'Δ':>10}")
    print("-" * 47)
    for model, computed in rows:
        ref = PUBLISHED.get(model)
        if ref is None:
            print(f"{model:<16}{computed:>10.5f}{'—':>11}{'—':>10}")
        else:
            print(f"{model:<16}{computed:>10.5f}{ref:>11.4f}{computed - ref:>+10.5f}")
    print("-" * 47)


if __name__ == "__main__":
    main()
