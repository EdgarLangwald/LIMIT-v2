"""Evaluation for LIMIT-v2: metrics, plotting, and interactive manual eval."""
import numpy as np
import matplotlib.pyplot as plt

from src.embed import embed_query


def evaluate(
    doc_embs:   np.ndarray,
    qry_embs:   np.ndarray,
    qrels:      list[list[int]],
    n_targets:  int,
    n_values:   list[int],
    ks:         list[int] = [1, 5, 10],
) -> dict[int, dict]:
    """Evaluate retrieval at increasing distractor pool sizes.

    For each n in n_values, the corpus is: all target docs (fixed) + first n distractor docs.
    Targets occupy indices 0..n_targets-1; qrels must only reference those indices.

    Returns dict mapping n -> {"recall@k": float, ..., "mrr": float}.
    """
    target_embs     = doc_embs[:n_targets]
    distractor_embs = doc_embs[n_targets:]

    # Zero-norm rows score 0 against every query; with strict-'>' ranking they all tie at
    # rank 1 and fake a perfect score. This is the signature of a corrupt/incomplete embedding
    # cache (e.g. an interrupted run), so fail loudly rather than report bogus metrics.
    n_zero_docs = int((np.linalg.norm(doc_embs, axis=1) < 1e-6).sum())
    if n_zero_docs:
        raise ValueError(
            f"{n_zero_docs}/{len(doc_embs)} doc embeddings are zero vectors — corpus embeddings "
            f"are corrupt or incomplete. Re-embed (delete the cached *_d.npy) before evaluating."
        )

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
            # rank of each relevant doc: 1 + number of corpus docs scoring strictly higher
            ranks = (q_scores[None, :] > rel_scores[:, None]).sum(axis=1) + 1

            mrr_vals.append(1.0 / float(ranks.min()))
            for k in ks:
                recall_hits[k].append(float((ranks <= k).sum() / len(rel)))

        out = {f"recall@{k}": float(np.mean(recall_hits[k])) for k in ks}
        out["mrr"] = float(np.mean(mrr_vals))
        results[n] = out

    return results


def plot_results(
    results: dict[int, dict],
    title:   str = "",
    show:    bool = True,
    show_mrr: bool = True,
) -> None:
    ns  = sorted(results)
    ks  = sorted(int(k.split("@")[1]) for k in results[ns[0]] if k.startswith("recall@"))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, k in enumerate(ks):
        ax.plot(ns, [results[n][f"recall@{k}"] for n in ns], marker="os^Dv"[i % 5], label=f"Recall@{k}")
    if show_mrr:
        ax.plot(ns, [results[n]["mrr"] for n in ns], marker="x", linestyle="--", label="MRR")

    ax.set_xlabel("n distractors")
    ax.set_ylabel("Score")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n/1000:.0f}k" if n >= 1000 else str(n) for n in ns])
    ax.set_ylim(0, 1.05)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        fname = f"{title}.png" if title else "results.png"
        plt.savefig(fname, dpi=150)
        plt.close(fig)


def plot_mrr_comparison(all_results: dict[str, dict[int, dict]], title: str = "", show: bool = True) -> None:
    """Compare MRR curves across models. all_results: {model_name: {n: metrics}}."""
    markers = "os^DvP*X"
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (model_name, results) in enumerate(all_results.items()):
        ns  = sorted(results)
        mrr = [results[n]["mrr"] for n in ns]
        ax.plot(ns, mrr, marker=markers[i % len(markers)], label=model_name)

    ns = sorted(next(iter(all_results.values())))
    ax.set_xlabel("n distractors")
    ax.set_ylabel("MRR")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n/1000:.0f}k" if n >= 1000 else str(n) for n in ns])
    ax.set_ylim(0, 1.05)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        fname = f"{title}.png" if title else "mrr_comparison.png"
        plt.savefig(fname, dpi=150)
        plt.close(fig)


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

    MODEL = "Snowflake_v2"
    N     = 1000
    M     = 4
    SEED  = 42
    K     = 10

    dataset_name = f"manual_n{N}_m{M}_s{SEED}"
    corpus_docs  = generate_corpus(N, M, seed=SEED)
    corpus       = {doc["name"]: " ".join(doc["sentences"]) for doc in corpus_docs}
    dataset      = {"corpus": corpus, "queries": {}}

    embs  = embed_dataset(dataset, MODEL, dataset_name)
    doc_embs = embs["doc_embs"]
    model = load_model(MODEL)

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
