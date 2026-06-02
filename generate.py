"""Dataset generation for LIMIT-v2 embedding stress tests."""
import json
import os
import random
import numpy as np
from faker import Faker
from profile_sentences import generate as _gen

# Build weighted sampling table once at import time.
# Excludes categories with frequency=0 (male_name, female_name, family_name, occupations).
_SAMPLING_NAMES: list[str] = []
_SAMPLING_CATS: list = []
_SAMPLING_WEIGHTS: list[float] = []

def _init_sampling():
    pairs = [
        (name, getattr(_gen, name))
        for name in vars(_gen)
        if not name.startswith('_')
    ]
    active = [(n, c) for n, c in pairs if c.frequency > 0]
    total = sum(c.frequency for _, c in active)
    _SAMPLING_NAMES.extend(n for n, _ in active)
    _SAMPLING_CATS.extend(c for _, c in active)
    _SAMPLING_WEIGHTS.extend(c.frequency / total for _, c in active)

_init_sampling()
_N_CATS = len(_SAMPLING_NAMES)


def generate_dataset(n: int, m: int, seed: int | None = None) -> list[dict]:
    """Generate n documents, each with an intro sentence plus m category-sampled sentences.

    Args:
        n: Number of person documents to generate.
        m: Number of category sentences sampled per document (not counting the intro).
        seed: Optional RNG seed for reproducibility.

    Returns:
        List of dicts with keys:
            name (str), gender (str), sentences (list[str]), facts (list[dict])
        facts[0] is the intro fact; facts[1:] correspond to sentences[1:].
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    Faker.seed(seed)
    fake = Faker()

    weights = np.array(_SAMPLING_WEIGHTS)
    replace = m > _N_CATS

    documents = []
    for _ in range(n):
        gender = 'male' if rng.random() < 0.5 else 'female'

        first = rng.choice(_gen.male_name.pool if gender == 'male' else _gen.female_name.pool)
        surname = rng.choice(_gen.family_name.pool)
        full_name = f"{first} {surname}"

        dob = fake.date_of_birth(minimum_age=15, maximum_age=90).strftime('%d %B %Y')
        state = fake.state()
        job = fake.job()
        intro = f"{full_name} was born on {dob}, lives in {state}, and works as a {job}."

        chosen = np_rng.choice(_N_CATS, size=m, replace=replace, p=weights)

        sentences = [intro]
        facts: list[dict] = [{"category": "intro", "name": full_name, "dob": dob, "state": state, "job": job}]

        for idx in chosen:
            cat_seed = rng.randint(0, 2**31)
            result = _SAMPLING_CATS[idx](seed=cat_seed, gender=gender)
            sentences.append(result["sentence"])
            facts.append(result["facts"])

        documents.append({"name": full_name, "gender": gender, "sentences": sentences, "facts": facts})

    return documents


if __name__ == "__main__":

    N=100
    M=4

    out_dir = os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(out_dir, exist_ok=True)

    filename = f"dataset_n{N}_m{M}.json"
    out_path = os.path.join(out_dir, filename)

    print(f"Generating {N} documents x {M} sentences (seed={42}) ...")
    docs = generate_dataset(N, M, seed=42)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(docs)} documents -> {out_path}")
