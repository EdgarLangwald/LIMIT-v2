import numpy as np
from src.evaluate import compute_score_matrix, recall_at_k  # TODO: metrics not yet implemented


def test_score_matrix():
    # Docs are basis vectors: cos(q, d_j) = q[j], so ranking = argsort of query coefficients
    n_docs = 5
    doc_embs = np.eye(n_docs, dtype=float)

    # Three queries with explicit similarity vectors — no ties, ranking fully determined
    qry_vecs = np.array([
        [1.0, 0.9, 0.8, 0.7, 0.6],  # ranking: 0, 1, 2, 3, 4
        [0.2, 0.5, 1.0, 0.7, 0.4],  # ranking: 2, 3, 1, 4, 0
        [0.8, 0.3, 0.6, 0.1, 0.9],  # ranking: 4, 0, 2, 1, 3
    ])
    qry_embs = qry_vecs / np.linalg.norm(qry_vecs, axis=1, keepdims=True)

    expected_rankings = np.argsort(-qry_vecs, axis=1)  # same order before and after normalization

    score_mat = compute_score_matrix(doc_embs, qry_embs)
    actual_rankings = np.argsort(-score_mat, axis=1)

    np.testing.assert_array_equal(actual_rankings, expected_rankings)


def test_recall():
    rng = np.random.default_rng(42)
    n_clusters = 4
    n_per_cluster = 5
    dim = 32
    kappa = 200  # high concentration → tight, well-separated clusters

    centers = rng.standard_normal((n_clusters, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    doc_embs_list = []
    for mu in centers:
        noise = rng.standard_normal((n_per_cluster, dim))
        samples = mu + noise / np.sqrt(kappa)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        doc_embs_list.append(samples)
    doc_embs = np.vstack(doc_embs_list)

    qry_embs = centers
    qrels = [list(range(i * n_per_cluster, (i + 1) * n_per_cluster)) for i in range(n_clusters)]

    # Verify the vMF construction actually separates clusters — if this fails, increase kappa
    scores = qry_embs @ doc_embs.T
    for q_idx in range(n_clusters):
        relevant = qrels[q_idx]
        irrelevant = [j for j in range(n_clusters * n_per_cluster) if j not in relevant]
        assert scores[q_idx, relevant].min() > scores[q_idx, irrelevant].max(), (
            f"vMF construction not well-separated for cluster {q_idx} — increase kappa"
        )

    assert recall_at_k(doc_embs, qry_embs, qrels, k=n_per_cluster) == 1.0
