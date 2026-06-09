from src.generate import generate_dataset


def test_dataset_metadata():
    dataset, qrels, n_targets = generate_dataset(n=10, m=2, seed=42)
    corpus  = dataset["corpus"]
    queries = dataset["queries"]

    assert len(corpus) == n_targets + 10, "corpus size should be n_targets + n fillers"
    for doc in corpus.values():
        assert isinstance(doc, str) and len(doc) > 0

    assert len(queries) > 0
    for q in queries.values():
        assert isinstance(q, str) and len(q) > 0

    assert len(qrels) == len(queries), "one qrel list per query"


def test_qrels():
    _, qrels, n_targets = generate_dataset(n=10, m=2, seed=42)

    for i, rel in enumerate(qrels):
        assert len(rel) > 0, f"query {i} has no relevant docs"
        for idx in rel:
            assert 0 <= idx < n_targets, (
                f"qrel index {idx} for query {i} out of target range [0, {n_targets})"
            )
