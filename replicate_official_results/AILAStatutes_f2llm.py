"""
Replicate the AILAStatutes MTEB benchmark for F2LLM-v2-1.7B via two approaches:

1. Manual  — dataset loaded directly from HuggingFace (pinned revision), queries encoded
   with the MTEB instruction template, metrics computed with the standard NDCG/Recall
   formula (mirrors what pytrec_eval produces inside MTEB).

2. MTEB    — uses mteb.get_model() + mteb.evaluate() as the official MTEB library.

Published reference (F2LLM-v2-1.7B revision 3766d46e, mteb 2.6.7):
    Dataset: mteb/AILA_statutes  revision ebfcd844eadd3d667efa3c57fc5c8c87f5c2867e
    Instruction (queries only): "Identify the most relevant statutes for the given situation."
    Instruction template:       "Instruct: {instruction}\\nQuery: "

    ndcg_at_10:   0.3336
    recall_at_1:  0.058
    recall_at_3:  0.141
    recall_at_5:  0.27167
    recall_at_10: 0.42867
    recall_at_20: 0.598

Run:
    PYTHONPATH=. python -c "
    from replicate_official_results.AILAStatutes_f2llm import replicate_aila_manual
    replicate_aila_manual()
    "
"""
import torch
from datasets import load_dataset

from replicate_official_results.replicate import _load
from replicate_official_results.AILAStatutes_qwen3 import _corpus_text, _compute_metrics

_DATASET_PATH = "mteb/AILA_statutes"
_DATASET_REVISION = "ebfcd844eadd3d667efa3c57fc5c8c87f5c2867e"
_TASK_INSTRUCTION = "Identify the most relevant statutes for the given situation."
_INSTRUCTION_PREFIX = f"Instruct: {_TASK_INSTRUCTION}\nQuery: "

# Published MTEB scores — used as reference in tests
PUBLISHED = {
    "ndcg_at_10":  0.3336,
    "recall_at_1":  0.058,
    "recall_at_3":  0.141,
    "recall_at_5":  0.27167,
    "recall_at_10": 0.42867,
    "recall_at_20": 0.598,
}


def _load_f2llm():
    return _load(
        "F2LLM_1_7b", "codefuse-ai/F2LLM-v2-1.7B",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )


def replicate_aila_manual(batch_size: int = 32) -> dict:
    """
    Direct replication via SentenceTransformer.encode():
    - Corpus documents: no prefix (apply_instruction_to_passages=False)
    - Queries: MTEB instruction template prepended
    Dataset pinned to the revision used for the published MTEB score.
    Returns a dict with recall_at_{1,3,5,10,20} and ndcg_at_10.
    """
    model = _load_f2llm()

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

    sim = query_embs @ corpus_embs.T
    metrics = _compute_metrics(query_ids, corpus_ids, qrels, sim)
    print(f"[aila_manual_f2llm] {metrics}")
    return metrics


def replicate_aila_mteb(batch_size: int = 32) -> dict:
    """
    Replication via mteb.get_model() + mteb.evaluate().
    Uses the MTEB-registered model config (correct instruction template, batch size, etc.).
    Results are cached in ~/.cache/mteb so subsequent calls are instant.
    Returns a dict with recall_at_{1,3,5,10,20} and ndcg_at_10.
    """
    import mteb

    model = mteb.get_model("codefuse-ai/F2LLM-v2-1.7B")
    task = mteb.get_task("AILAStatutes")
    model_result = mteb.evaluate(model=model, tasks=[task], encode_kwargs={"batch_size": batch_size})

    task_result = model_result.task_results[0]
    scores_list = task_result.scores.get("test", [{}])
    scores = scores_list[0] if scores_list else {}

    keys = ["ndcg_at_10", "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "recall_at_20"]
    metrics = {k: scores.get(k) for k in keys}
    print(f"[aila_mteb_f2llm] {metrics}")
    return metrics
