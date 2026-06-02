## Overview

LIMIT-v2 is the second version of a project with the purpose of stress testing embedding models. The first version sits next to this project, i.e. at 
../LIMIT

## Design Philosophy

All functions in the code must fit into ONLY ONE of three categories:

Dataset creation -> Embedding -> Evaluation. 

This way, multiple tests can be done on multiple models. The embedding has to be able to run locally and on a cluster. And different metrics have to be able to evaluate performance without recomputing embeddings every time.


## Project structure

- `explore_faker.ipynb` — exploratory notebook for the Faker library
- `build_corpus_generators.py` — calls Claude Haiku per JSON in dariusk/corpora to evaluate and generate sentence-family dataclasses; writes `profile_sentences.py`
- `corpus_evaluations/` — intermediate per-file results (resumable cache, not checked in)
- `profile_sentences.py` — auto-generated output: one dataclass per accepted corpora category; also contains dataclasses for male/female first names and family surnames (loaded from CSV at import time)
- `male_names.csv`, `female_names.csv`, `family_names.csv` — name pools loaded by `profile_sentences.py` at import time (not checked in to avoid 150k-row bloat)
- `generate.py` — dataset creation: `generate_dataset(n, m, seed)` builds n person documents each with a Faker intro sentence plus m category-sampled sentences; returns list of dicts with `name`, `gender`, `sentences`, `facts`. Run directly with `--n`, `--m`, `--seed`, `--out` flags.
- `dataset/` — generated JSON files from `generate.py` (not checked in)

### Using `profile_sentences.generate`

The module exposes a single `_Generate` instance called `generate`. Every attribute on it is a callable dataclass that returns a `{"sentence": ..., "facts": {"category": ..., "item": ...}}` dict:

```python
from profile_sentences import generate

generate.vegetables(seed=42)                     # male (default)
generate.vegetables(seed=42, gender='female')    # female
```

`gender` accepts `'male'` (default) or `'female'`. Pronouns in templates (`{He}`, `{his}`, `{him}`, `{himself}`, etc.) are resolved by a `_PRONOUNS` dict at call time. `{HisP}`/`{hisP}` resolve to "his"/"hers" (possessive pronoun, not adjective).

Each category dataclass also exposes read-only metadata:

```python
generate.vegetables.pool        # list of items
generate.vegetables.pool_size   # len(pool)
generate.vegetables.templates   # list of template strings
generate.vegetables.frequency   # 0–100, % likelihood this appears in a profile
generate.vegetables.description
generate.vegetables.category_name
```

**All 114 categories** are listed as fields on `_Generate` in [profile_sentences.py](profile_sentences.py). To see them all: `[f for f in vars(generate)]`.

**You have to keep this relevant**. Every time you delete files or create new files that are relevant and meant to stay, update this (no throwaway or miscellanious files)

## Coding practices

**IMPORTANT:** Use the correct venv based on the machine name. To get it, run `hostname` in bash:
- `EDGAR-PC`: `C:\Users\edgar\Projekte\Python\Machine_Learning_venv\Scripts\python.exe`
- `EDGAR_LAPTOP`: `C:\Users\EdgarLangwald\OneDrive - neuland AI AG\Coding\.venv\Scripts\python.exe`
- `*.rwth-aachen.de` (RWTH cluster): `/rwthfs/rz/cluster/home/nld68820/.venv/bin/python`

Run all scripts and pip installs using the correct python for the current device.

## Other

EDGAR-PC and EDGAR_LAPTOP run on windows. Keep in mind that spaces in paths compicate the bash path finding.

