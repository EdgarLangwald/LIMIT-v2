"""Evaluation for LIMIT-v2: metrics and JSON persistence."""
import json
from pathlib import Path

import numpy as np

from src.embed import embed_query
from src.paths import RESULTS_DIR


def evaluate(
    embs:        dict,
    qrels:       list[list[int]],
    n_targets:   int,
    embs_path:   Path,
    n_values:    list[int] | None = None,
    ks:          list[int] = [1, 5, 10],
    results_dir: Path | None = None,
    force:       bool = True,
) -> dict[int, dict]:
    """Evaluate retrieval at increasing distractor pool sizes.

    For each n in n_values, the corpus is: all target docs (fixed) + first n
    distractor docs.  Targets occupy indices 0..n_targets-1; qrels must only
    reference those indices.

    If *embs_path* is given (as returned by embed_dataset), results are saved to
    <results_dir>/<model>_<dataset>.json.  With force=False an existing file is
    loaded and returned immediately.

    Returns dict mapping n -> {"recall@k": float, ..., "mrr": float}.
    """
    n_values = [n_targets] if n_values is None else n_values
    results_dir = results_dir or RESULTS_DIR

    name = None
    if embs_path is not None:
        embs_path = Path(embs_path)
        name = f"{embs_path.name}_{embs_path.parent.name}"

    if name is not None:
        json_path = results_dir / f"{name}.json"
        if not force and json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            return {int(k): v for k, v in data["results"].items()}

    doc_embs = embs["doc_embs"]
    qry_embs = embs["qry_embs"]

    # Zero-norm rows score 0 against every query; with strict-'>' ranking they
    # all tie at rank 1 and fake a perfect score — fail loudly instead.
    n_zero_docs = int((np.linalg.norm(doc_embs, axis=1) < 1e-6).sum())
    if n_zero_docs:
        raise ValueError(
            f"{n_zero_docs}/{len(doc_embs)} doc embeddings are zero vectors — "
            f"corpus embeddings are corrupt or incomplete. Re-embed (delete the "
            f"cached *_d.npy) before evaluating."
        )

    target_embs     = doc_embs[:n_targets]
    distractor_embs = doc_embs[n_targets:]

    results = {}
    for n in n_values:
        corpus = np.concatenate([target_embs, distractor_embs[:n]], axis=0)
        scores = qry_embs @ corpus.T  # (n_queries, n_corpus)

        mrr_vals    = []
        recall_hits = {k: [] for k in ks}

        for qi, rel in enumerate(qrels):
            q_scores   = scores[qi]
            rel_arr    = np.array(rel)
            rel_scores = q_scores[rel_arr]
            ranks = (q_scores[None, :] > rel_scores[:, None]).sum(axis=1) + 1

            mrr_vals.append(1.0 / float(ranks.min()))
            for k in ks:
                recall_hits[k].append(float((ranks <= k).sum() / len(rel)))

        out = {f"recall@{k}": float(np.mean(recall_hits[k])) for k in ks}
        out["mrr"] = float(np.mean(mrr_vals))
        results[n] = out

    if name is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": name,
            "ks": ks,
            "results": {str(n): v for n, v in results.items()},
        }
        with open(results_dir / f"{name}.json", "w") as f:
            json.dump(payload, f, indent=2)

    return results


def evaluate_manually(
    query:       str,
    target_name: str,
    doc_embs:    np.ndarray,
    corpus:      dict[str, str],
    model_name:  str,
    k:           int = 10,
    model=None,
) -> dict:
    """Embed query, rank corpus by cosine similarity, return top-k and target rank."""
    names     = list(corpus.keys())
    query_emb = embed_query(query, model_name, model)
    sims      = doc_embs @ query_emb
    ranked    = np.argsort(sims)[::-1]

    top_k = [(i + 1, names[idx]) for i, idx in enumerate(ranked[:k])]
    target_lower = target_name.lower()
    target_rank  = next(
        (r for r, idx in enumerate(ranked, 1) if names[idx].lower() == target_lower),
        None,
    )
    return {"top_k": top_k, "target_rank": target_rank}


if __name__ == "__main__":
    import os
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.isfile(env_path):
        for _line in open(env_path):
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

    from src.generate import generate_corpus
    from src.embed import embed_dataset, load_model
    from src.paths import GENERATED_DATASETS_DIR

    MODEL = "Snowflake_v2"
    N     = 1000
    M     = 4
    SEED  = 42
    K     = 10

    corpus_docs  = generate_corpus(N, M, seed=SEED)
    corpus       = {doc["name"]: " ".join(doc["sentences"]) for doc in corpus_docs}
    dataset      = {"corpus": corpus, "queries": {}}
    dataset_path = GENERATED_DATASETS_DIR / f"manual_n{N}_m{M}_s{SEED}.json"

    embs, _  = embed_dataset(dataset, MODEL, dataset_path)
    doc_embs = embs["doc_embs"]
    model    = load_model(MODEL)

    print(f"Corpus: {len(corpus)} docs. Top-{K} results. Type 'quit' to exit.\n")
    while True:
        query = input("Query  : ").strip()
        if query.lower() in ("quit", "q", "exit", ""):
            break
        target = input("Target : ").strip()
        if not target:
            continue

        result = evaluate_manually(query, target, doc_embs, corpus, MODEL, k=K, model=model)

        for rank, name in result["top_k"]:
            marker = "  <--" if name.lower() == target.lower() else ""
            print(f"  {rank:>3}. {name}{marker}")

        if result["target_rank"] is not None:
            print(f"\n  Rank: {result['target_rank']} / {len(corpus)}\n")
        else:
            print(f"\n  '{target}' not found\n")
