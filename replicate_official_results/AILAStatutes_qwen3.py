"""
Replicate the AILAStatutes MTEB benchmark for Qwen3-Embedding-0.6B via two approaches:

1. Manual  — dataset loaded directly from HuggingFace (pinned revision), queries encoded
   with the MTEB instruction template, metrics computed with the standard NDCG/Recall
   formula (mirrors what pytrec_eval produces inside MTEB).

2. MTEB    — uses mteb.get_model() + mteb.evaluate() as the official MTEB library.
   Note: installed mteb version (2.x) may differ from the published run (1.38.9), so
   small numeric deltas are expected.

Published reference (Qwen3-Embedding-0.6B revision b22da495, mteb 1.38.9):
    Dataset: mteb/AILA_statutes  revision ebfcd844eadd3d667efa3c57fc5c8c87f5c2867e
    Instruction (queries only): "Identifying the most relevant statutes for a given situation"
    Instruction template:       "Instruct: {instruction}\\nQuery: "

    ndcg_at_10:   0.79018
    recall_at_1:  0.20933
    recall_at_3:  0.46467
    recall_at_5:  0.67333
    recall_at_10: 0.854
    recall_at_20: 0.957

Run:
    PYTHONPATH=. python -c "
    from replicate_official_results.AILAStatutes import replicate_aila_manual
    replicate_aila_manual()
    "
"""
import math

import numpy as np
import torch
from datasets import load_dataset

from replicate_official_results.replicate import _load

_DATASET_PATH = "mteb/AILA_statutes"
_DATASET_REVISION = "ebfcd844eadd3d667efa3c57fc5c8c87f5c2867e"
_TASK_INSTRUCTION = "Identifying the most relevant statutes for a given situation"
_INSTRUCTION_PREFIX = f"Instruct: {_TASK_INSTRUCTION}\nQuery:"

# Published MTEB scores — used as reference in tests
PUBLISHED = {
    "ndcg_at_10": 0.79018,
    "recall_at_1": 0.20933,
    "recall_at_3": 0.46467,
    "recall_at_5": 0.67333,
    "recall_at_10": 0.854,
    "recall_at_20": 0.957,
}


def _load_qwen3():
    return _load(
        "Qwen0.6b", "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
    )


def _corpus_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return (title + " " + text).strip() if title else text


def _compute_metrics(
    query_ids: list,
    corpus_ids: list,
    qrels: dict,
    sim_matrix: np.ndarray,
    ks: tuple = (1, 3, 5, 10, 20),
) -> dict:
    """Recall@k and NDCG@10 using the standard BEIR/MTEB per-query formula."""
    retrieve_k = max(max(ks), 10)
    recall: dict[int, list] = {k: [] for k in ks}
    ndcg: list[float] = []

    for qi, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        rel = {cid: s for cid, s in qrels[qid].items() if s > 0}
        if not rel:
            continue

        top_indices = np.argsort(-sim_matrix[qi])[:retrieve_k]
        top_ids = [corpus_ids[j] for j in top_indices]
        n_rel = len(rel)

        for k in ks:
            hits = sum(1 for cid in top_ids[:k] if cid in rel)
            recall[k].append(hits / n_rel)

        # NDCG@10: DCG / IDCG with gain = relevance score
        dcg = sum(
            rel.get(cid, 0) / math.log2(rank + 2)
            for rank, cid in enumerate(top_ids[:10])
        )
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
        ndcg.append(dcg / idcg if idcg else 0.0)

    result = {f"recall_at_{k}": float(np.mean(recall[k])) for k in ks}
    result["ndcg_at_10"] = float(np.mean(ndcg))
    return result


def replicate_aila_manual(batch_size: int = 32) -> dict:
    """
    Direct replication via SentenceTransformer.encode():
    - Corpus documents: no prefix (apply_instruction_to_passages=False)
    - Queries: MTEB instruction template prepended
    Dataset pinned to the revision used for the published MTEB score.
    Returns a dict with recall_at_{1,3,5,10,20} and ndcg_at_10.
    """
    model = _load_qwen3()

    corpus_ds = load_dataset(_DATASET_PATH, name="corpus", revision=_DATASET_REVISION, split="corpus")
    queries_ds = load_dataset(_DATASET_PATH, name="queries", revision=_DATASET_REVISION, split="queries")
    qrels_ds = load_dataset(_DATASET_PATH, name="default", revision=_DATASET_REVISION, split="test")

    corpus_ids = list(corpus_ds["_id"])
    corpus_texts = [_corpus_text(row) for row in corpus_ds]

    query_ids = list(queries_ds["_id"])
    query_texts = list(queries_ds["text"])

    qrels: dict[str, dict[str, float]] = {}
    for row in qrels_ds:
        qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = row["score"]

    paired = [(qid, qt) for qid, qt in zip(query_ids, query_texts) if qid in qrels]
    query_ids = [p[0] for p in paired]
    query_texts = [p[1] for p in paired]

    with torch.no_grad():
        corpus_embs = model.encode(
            corpus_texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True,
        )
        query_embs = model.encode(
            query_texts, prompt=_INSTRUCTION_PREFIX, batch_size=batch_size,
            show_progress_bar=True, normalize_embeddings=True,
        )

    sim = query_embs @ corpus_embs.T  # cosine similarity (embeddings are normalized)
    metrics = _compute_metrics(query_ids, corpus_ids, qrels, sim)
    print(f"[aila_manual] {metrics}")
    return metrics


def replicate_aila_mteb(batch_size: int = 32) -> dict:
    """
    Replication via mteb.get_model() + mteb.evaluate().
    Uses the MTEB-registered model config (correct instruction template, batch size, etc.).
    Results are cached in ~/.cache/mteb so subsequent calls are instant.
    Returns a dict with recall_at_{1,3,5,10,20} and ndcg_at_10.
    """
    import mteb

    model = mteb.get_model("Qwen/Qwen3-Embedding-0.6B")
    task = mteb.get_task("AILAStatutes")
    model_result = mteb.evaluate(model=model, tasks=[task], encode_kwargs={"batch_size": batch_size})

    task_result = model_result.task_results[0]
    # scores: dict[split_name, list[ScoresDict]]
    scores_list = task_result.scores.get("test", [{}])
    scores = scores_list[0] if scores_list else {}

    keys = ["ndcg_at_10", "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "recall_at_20"]
    metrics = {k: scores.get(k) for k in keys}
    print(f"[aila_mteb] {metrics}")
    return metrics
