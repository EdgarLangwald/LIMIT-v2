"""STAGE 1.5: corpus distortions for stress-testing embedding robustness.

Each function takes a dataset (the dict returned by ``generate_dataset``:
``{"corpus": {...}, "queries": {...}}``) and returns a new dataset with the
**corpus** perturbed and the **queries left untouched**. The corpus holds both
the target docs (``doc_0 … doc_{n_targets-1}``) and the distractors, so a single
pass distorts everything we retrieve over without touching what we search with.

Usage:
    dataset = add_random_chars(dataset, density=0.1, replace=False)
"""
import json
import random
import string

from src.paths import DATASET_DIR

_ALPHABET = string.ascii_lowercase
_MAX_DENSITY = 0.5  # density is the random-char fraction of the final text; >0.5 is meaningless


def _sample_density(mean: float, spread: float, rng: random.Random) -> float:
    """Draw a per-document density from Gaussian(mean, spread), clamped to [0, _MAX_DENSITY]."""
    d = rng.gauss(mean, spread) if spread > 0 else mean
    return max(0.0, min(d, _MAX_DENSITY))


def _distort_text(text: str, density: float, replace: bool, rng: random.Random) -> str:
    """Perturb one document so that random chars make up ``density`` of the result.

    density == (# random chars) / (# total chars) after the operation.
      - replace=True:  overwrite round(density * L) of the L original chars with random ones.
                       Result length stays L; random/total == density exactly (modulo rounding).
      - replace=False: insert r random chars, where r/(L + r) == density, i.e.
                       r = round(density * L / (1 - density)). Original chars are all kept;
                       random/total -> density. (Can never reach 1, hence the 0.5 cap upstream.)
    """
    L = len(text)
    if L == 0 or density <= 0:
        return text

    chars = list(text)
    if replace:
        k = min(round(density * L), L)
        for i in rng.sample(range(L), k):
            chars[i] = rng.choice(_ALPHABET)
    else:
        r = round(density * L / (1 - density))
        for _ in range(r):
            chars.insert(rng.randint(0, len(chars)), rng.choice(_ALPHABET))
    return "".join(chars)


def _profile_block(text: str) -> str:
    """Render one distorted doc with each sentence on its own line (sentences are '. '-separated)."""
    return text.replace(". ", ".\n")


def add_random_chars(
    dataset: dict,
    density: float = 0.1,
    replace: bool = False,
    density_spread: float = 0.0,
    seed: int | None = 42,
    save: str | None = None,
    n_targets: int | None = None,
) -> dict:
    """Insert/replace random English letters into every corpus document.

    Args:
        dataset:        {"corpus": {doc_id: text, ...}, "queries": {...}} from generate_dataset.
        density:        mean target ratio of random chars to total chars per distorted doc,
                        in [0, 0.5]. Capped at 0.5 (values above are clamped).
        replace:        True  -> replace `density` fraction of existing chars with random ones.
                        False -> insert random chars until random/total == density (docs grow).
        density_spread: std of the per-document density. The density actually applied to each doc
                        is drawn from Gaussian(density, density_spread) clamped to [0, 0.5].
                        0 (default) -> every doc uses `density` exactly.
        seed:           RNG seed for reproducible distortion.
        save:           None (default) | "txt" (human-readable preview) | "json" (full dataset dict).
        n_targets:      Number of leading target docs in the corpus (doc_0..doc_{n_targets-1}).
                        Used only by save="txt": the preview writes all fillers, a "-----" delimiter,
                        then a random sample of that-many target profiles (targets are too numerous
                        to dump in full). If None, the txt dumps the whole corpus.

    Returns:
        A new dataset dict with a distorted "corpus" and the original "queries".
    """
    density = min(density, _MAX_DENSITY)
    rng = random.Random(seed)
    new_corpus = {
        doc_id: _distort_text(text, _sample_density(density, density_spread, rng), replace, rng)
        for doc_id, text in dataset["corpus"].items()
    }
    distorted = {"corpus": new_corpus, "queries": dataset["queries"]}

    if save is not None:
        stem = f"distorted_d{density}_sp{density_spread}_r{int(replace)}_s{seed}"
        if save == "txt":
            docs = list(new_corpus.values())
            split = n_targets or 0
            targets, fillers = docs[:split], docs[split:]
            blocks = ["\n\n".join(_profile_block(d) for d in fillers)]
            if targets:  # sample as many targets as there are fillers (full dump is too large)
                k = min(len(fillers), len(targets)) or len(targets)
                sample = random.Random(seed).sample(targets, k)
                blocks.append("\n\n".join(_profile_block(d) for d in sample))
            path = str(DATASET_DIR / f"{stem}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n\n-----\n\n".join(blocks))
        elif save == "json":
            path = str(DATASET_DIR / f"{stem}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(distorted, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"save must be 'txt', 'json', or None — got {save!r}")

    return distorted
