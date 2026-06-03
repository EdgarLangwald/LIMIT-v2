"""Manual interactive evaluation for LIMIT-v2."""
import numpy as np
from src.embed import embed_query


def evaluate_manually(
    query: str,
    target_name: str,
    doc_embs: np.ndarray,
    docs: list[dict],
    model_name: str,
    k: int = 10,
    model=None,
) -> dict:
    """Embed query, rank all documents by cosine similarity, return top-k and target rank.

    Args:
        query:       Natural language query string.
        target_name: Full name of the target person (matched case-insensitively against doc['name']).
        doc_embs:    (n, dim) float32 L2-normalized document embeddings from embed_dataset().
        docs:        Document list from generate_dataset() — same order as doc_embs.
        model_name:  Key into MODELS dict.
        k:           Number of top results to return.
        model:       Pre-loaded model object (required; use embed.load_model() before calling).

    Returns:
        {
            "top_k":       [(rank, name), ...],   # length min(k, n)
            "target_rank": int | None,            # 1-based rank, or None if name not found
        }
    """
    query_emb = embed_query(query, model_name, model)  # (dim,)
    sims      = doc_embs @ query_emb                   # (n,) cosine similarities

    ranked_indices = np.argsort(sims)[::-1]
    names_ranked   = [docs[idx]["name"] for idx in ranked_indices]

    top_k = [(i + 1, name) for i, name in enumerate(names_ranked[:k])]

    target_lower = target_name.lower()
    target_rank  = next(
        (rank for rank, name in enumerate(names_ranked, start=1) if name.lower() == target_lower),
        None,
    )

    return {"top_k": top_k, "target_rank": target_rank}
