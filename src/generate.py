"""Dataset generation for LIMIT-v2 embedding stress tests."""
import json
import os
import random
import numpy as np
from faker import Faker
from src.profile_sentences import generate as _gen
from src.paths import DATASET_DIR, EVAL_TARGETS

_SAMPLING_NAMES: list[str] = []
_SAMPLING_CATS: list = []
_SAMPLING_WEIGHTS: list[float] = []

def _init_sampling():
    pairs = [(name, getattr(_gen, name)) for name in vars(_gen) if not name.startswith('_')]
    active = [(n, c) for n, c in pairs if c.frequency > 0]
    total = sum(c.frequency for _, c in active)
    _SAMPLING_NAMES.extend(n for n, _ in active)
    _SAMPLING_CATS.extend(c for _, c in active)
    _SAMPLING_WEIGHTS.extend(c.frequency / total for _, c in active)

_init_sampling()
_N_CATS = len(_SAMPLING_NAMES)


def generate_corpus(n: int, m: int, seed: int | None = None, save: str | None = None) -> list[dict]:
    """Generate n filler documents, each with an intro sentence plus m category sentences.

    Args:
        n:    Number of person documents.
        m:    Category sentences per document (not counting intro).
        seed: Optional RNG seed.
        save: None (default) | "txt" (human-readable) | "json" (cached list for reuse).

    Returns:
        List of dicts with keys: name (str), gender (str), sentences (list[str]).
    """
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    Faker.seed(seed)
    fake   = Faker()

    weights = np.array(_SAMPLING_WEIGHTS)
    replace = m > _N_CATS

    documents = []
    for _ in range(n):
        gender = 'male' if rng.random() < 0.5 else 'female'

        first   = rng.choice(_gen.male_name.pool if gender == 'male' else _gen.female_name.pool)
        surname = rng.choice(_gen.family_name.pool)
        name    = f"{first} {surname}"

        dob   = fake.date_of_birth(minimum_age=15, maximum_age=90).strftime('%d %B %Y')
        state = fake.state()
        job   = fake.job()
        intro = f"{name} was born on {dob}, lives in {state}, and works as a {job}."

        chosen    = np_rng.choice(_N_CATS, size=m, replace=replace, p=weights)
        sentences = [intro]
        for idx in chosen:
            cat_seed = rng.randint(0, 2**31)
            sentences.append(_SAMPLING_CATS[idx](seed=cat_seed, gender=gender)["sentence"])

        documents.append({"name": name, "gender": gender, "sentences": sentences})

    if save is not None:
        stem = f"corpus_n{n}_m{m}_s{seed}"
        if save == "txt":
            path = str(DATASET_DIR / f"{stem}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(" ".join(doc["sentences"]) for doc in documents))
        elif save == "json":
            path = str(DATASET_DIR / f"{stem}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"save must be 'txt', 'json', or None — got {save!r}")

    return documents


def generate_dataset(
    n: int,
    m: int,
    seed: int | None = None,
) -> tuple[dict, list[list[int]], int]:
    """Build the full evaluation dataset: benchmark targets + n filler documents.

    Loads eval_targets.json, prepends unique target docs to an n-doc filler corpus,
    then builds aligned queries and qrels from the benchmark draws.

    Args:
        n:    Number of filler (distractor) documents.
        m:    Category sentences per filler document.
        seed: RNG seed for filler generation.

    Returns:
        dataset:   {"corpus": {"doc_0": text, ...}, "queries": {"query_0": text, ...}}
        qrels:     list[list[int]] — relevant doc indices per query (always in 0..n_targets-1)
        n_targets: number of target documents (always at the front of corpus)
    """
    with open(EVAL_TARGETS, encoding="utf-8") as f:
        draws = json.load(f)

    # Collect unique target docs preserving first-occurrence order
    seen = set()
    target_docs = []
    for draw in draws:
        for t in draw["targets"]:
            if t["name"] not in seen:
                seen.add(t["name"])
                target_docs.append(t)

    n_targets = len(target_docs)

    # Build corpus: targets first (indices 0..n_targets-1), fillers after
    corpus: dict[str, str] = {}
    for i, t in enumerate(target_docs):
        corpus[f"doc_{i}"] = " ".join(t["sentences"])

    for i, doc in enumerate(generate_corpus(n, m, seed)):
        corpus[f"doc_{n_targets + i}"] = " ".join(doc["sentences"])

    # Index map for fast qrels lookup
    name_to_idx = {t["name"]: i for i, t in enumerate(target_docs)}

    queries: dict[str, str] = {}
    qrels:   list[list[int]] = []
    qid = 0
    for draw in draws:
        target_indices = [name_to_idx[t["name"]] for t in draw["targets"]]
        queries[f"query_{qid}"]     = draw["query_specific"]
        qrels.append(target_indices)
        qid += 1
        queries[f"query_{qid}"]     = draw["query_broad"]
        qrels.append(target_indices)
        qid += 1

    assert len(queries) == len(qrels)
    assert all(all(0 <= r < n_targets for r in rel) for rel in qrels)

    return {"corpus": corpus, "queries": queries}, qrels, n_targets


if __name__ == "__main__":
    N, M, SEED = 100, 4, 42
    print(f"Generating corpus: n={N}, m={M}, seed={SEED}")
    generate_corpus(N, M, seed=SEED, save="json")
    print("Done.")
