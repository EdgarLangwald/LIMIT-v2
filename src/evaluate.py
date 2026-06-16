"""Evaluation for LIMIT-v2: metrics and JSON persistence."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.embed import embed_query
from src.paths import RESULTS_DIR


def evaluate(
    embs:           dict,
    qrels:          list[list[int]],
    n_targets:      int,
    embs_path:      Path,
    query_types:    list[str],
    n_values:       list[int] | None = None,
    ks:             list[int] = [1, 5, 10],
    results_dir:    Path | None = None,
    force:          bool = True,
    doc_batch_size: int = 2048,
    device:         str | None = None,
) -> dict[int, dict]:
    """Evaluate retrieval at increasing distractor pool sizes, separated by query type.

    Corpus at each n: all n_targets target docs + first n distractor docs.
    Uses an incremental rank-counting approach (no full score matrix in RAM):
    for each sorted n, only the newly added distractors are scored. Assumes
    exactly len(qrels[0]) relevant docs per query and that all targets are
    always present, so no per-query validity filtering is needed.

    query_types[i] is the type of query i (aligned with qrels and the query embeddings).
    Metrics are NOT pooled across types: every metric is a list whose j-th entry is the
    score over the queries of type ``type_order[j]``, where ``type_order`` is the de-duplicated
    first-occurrence order of query_types (also written to the JSON under "query_types").

    Saves JSON after each n checkpoint; returns immediately if all n_values are already
    present in the JSON (with matching query types) and force=False.
    """
    n_values    = [n_targets] if n_values is None else n_values
    results_dir = results_dir or RESULTS_DIR
    type_order  = list(dict.fromkeys(query_types))   # canonical index legend for the lists

    name      = None
    json_path = None
    if embs_path is not None:
        embs_path = Path(embs_path)
        name      = f"{embs_path.name}_{embs_path.parent.name}"
        json_path = results_dir / f"{name}.json"

    if not force and json_path is not None and json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        saved = {int(k): v for k, v in data["results"].items()}
        if all(n in saved for n in n_values) and data.get("query_types") == type_order:
            return saved

    doc_embs  = embs["doc_embs"]
    qry_embs  = embs["qry_embs"]
    n_queries = len(qrels)
    n_rel     = len(qrels[0])

    # query_types is aligned with qrels and the query embeddings by construction (it is
    # derived from the query keys in generate_dataset). Assert it loudly so any drift —
    # e.g. a partially-written embedding cache — fails here rather than silently mis-grouping.
    if not (len(query_types) == n_queries == int(np.asarray(qry_embs).shape[0])):
        raise ValueError(
            f"length mismatch: query_types={len(query_types)}, qrels={n_queries}, "
            f"qry_embs={int(np.asarray(qry_embs).shape[0])} — refusing to group by type"
        )

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # All query vectors fit in GPU memory comfortably
    q_vecs = torch.from_numpy(np.asarray(qry_embs)).to(dev)  # (n_queries, dim)

    # Row indices of the queries belonging to each type, in type_order. Grouping is by this
    # explicit label — never by position modulo len(type_order).
    type_rows = [
        (t, torch.tensor([i for i, qt in enumerate(query_types) if qt == t],
                         dtype=torch.long, device=dev))
        for t in type_order
    ]

    # Pre-compute relevant-doc scores once — (n_queries, n_rel)
    rel_idx  = np.array(qrels, dtype=np.int64)                # (n_queries, n_rel)
    rel_docs = torch.from_numpy(
        np.asarray(doc_embs[rel_idx.ravel()]).reshape(n_queries, n_rel, -1)
    ).to(dev)
    r_scores = torch.bmm(rel_docs, q_vecs.unsqueeze(-1)).squeeze(-1)  # (n_queries, n_rel)
    del rel_docs

    # better[qi, ri] = number of corpus docs scoring strictly above rel doc ri for query qi
    better = torch.zeros((n_queries, n_rel), dtype=torch.int64, device=dev)

    def _accumulate(start: int, end: int, emb_array) -> None:
        for s in range(start, end, doc_batch_size):
            chunk = torch.from_numpy(np.asarray(emb_array[s:min(s + doc_batch_size, end)])).to(dev)
            scores = q_vecs @ chunk.T                                       # (n_queries, chunk)
            better.add_((scores[:, None, :] - 1e-6 > r_scores[:, :, None]).sum(dim=2))

    # Score all targets (always in corpus) before the n loop
    t0 = time.time()
    _accumulate(0, n_targets, doc_embs)
    print(f"  [targets] scored {n_targets} docs in {time.time()-t0:.1f}s")

    distractor_embs = doc_embs[n_targets:]
    n_prev   = 0
    results  = {}

    for n in sorted(n_values):
        t_n    = time.time()
        n_new  = min(n, len(distractor_embs))
        _accumulate(n_prev, n_new, distractor_embs)
        n_prev = n_new

        ranks = better + 1                                                  # (n_queries, n_rel)
        best  = ranks.min(dim=1).values.float()                            # best rank per query
        # Each metric is a list over type_order: the score within that type's queries only.
        out = {
            f"recall@{k}": [float((ranks[rows] <= k).float().mean()) for _, rows in type_rows]
            for k in ks
        }
        out["mrr"] = [float((1.0 / best[rows]).mean()) for _, rows in type_rows]
        results[n] = out

        r0 = out[f"recall@{ks[0]}"]
        per_type = "  ".join(f"{t}={v:.3f}" for (t, _), v in zip(type_rows, r0))
        print(f"  [n={n:>9,}] recall@{ks[0]} [{per_type}]  ({time.time()-t_n:.1f}s)")

        if json_path is not None:
            results_dir.mkdir(parents=True, exist_ok=True)
            payload = {"name": name, "ks": ks, "query_types": type_order,
                       "results": {str(k): v for k, v in results.items()}}
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2)

    print(f"  Evaluation complete.")
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
