## Overview

LIMIT-v2 is the second version of a project with the purpose of stress testing embedding models. The first version sits
next to this project, i.e. at 
../LIMIT

## Design Philosophy

The codebase is built in roughly three stages:

Dataset creation -> Embedding -> Evaluation. 

This way, the building logic of the synthetic dataset is clearly separated from evaluation of the models. The embedding has to be able to run locally and on a cluster. And different metrics have to be able to evaluate performance without recomputing embeddings every time.


## Project structure

`src/` is the importable package for the three pipeline stages. `dataset/` contains both the build scripts and the generated data. Run all commands from the repo root.

```
LIMIT-v2/
├── main.py                       — full-pipeline entrypoint
├── src/                          — importable package (the three pipeline stages)
│   ├── paths.py                  — repo-root-anchored data dirs (DATASET_DIR, EMBEDDINGS_DIR, …)
│   ├── profile_sentences.py      — STAGE 1: sentence category dataclasses (auto-generated)
│   ├── generate.py               — STAGE 1: generate_dataset(n, m, seed)
│   ├── embed.py                  — STAGE 2: embed_dataset, load_model, embed_query
│   ├── evaluate.py               — STAGE 3: evaluate() saves results/{name}.json; evaluate_manually()
│   ├── plot.py                   — STAGE 3: visualize_results() builds PDF from results/*.json
│   └── pools/                    — male_names.csv, female_names.csv, family_names.csv, eval_targets.json
├── tests/                        — test scaffolds (some reference not-yet-built metrics)
├── dataset/
│   ├── corpus_workflow.js        — Workflow script: rebuilds src/profile_sentences.py
│   ├── corpus_workflow.txt       — subagent prompt for corpus_workflow.js
│   ├── corpus_handpicked_frequencies.txt — manually-set frequency weights for all 114 categories
│   ├── build_query_workflow_input.py — rebuilds src/pools/eval_targets.json benchmark
│   ├── query_workflow.js         — Workflow script: LLM step for query generation
│   ├── query_workflow.txt        — subagent prompt for query_workflow.js
│   ├── corpus_evaluations/       — per-file Haiku eval cache for corpus_workflow.js
│   ├── query_workflow_input/  query_workflow_output/ — intermediate query-build cache
│   └── generated_datasets/       — cached generate_dataset() outputs (n{n}_m{m}_s{seed}.json)
├── embeddings/ models/           — cached embeddings and model weights
├── results/                      — evaluate() JSON outputs + report.pdf from visualize_results()
└── explore_faker.ipynb
```

The tree HAS to be kept updated with PERMANENT files. If you create a file, make a judgement on wether the users intent is to keep the file in the long run, or whether the file's purpose is temporary (bugfix, exploration, etc.) 
IF UNSURE, ASK!

### Using `profile_sentences.generate`

The module exposes a single `_Generate` instance called `generate`. Every attribute on it is a callable dataclass that returns a `{"sentence": ..., "facts": {"category": ..., "item": ...}}` dict:

```python
from src.profile_sentences import generate

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

**All 114 categories** are listed as fields on `_Generate` in [src/profile_sentences.py](src/profile_sentences.py). To see them all: `[f for f in vars(generate)]`.

## Coding practices

**IMPORTANT:** Use the correct venv based on the machine name. To get it, run `hostname` in bash:
- `EDGAR-PC`: `C:\Users\edgar\Projekte\Python\Machine_Learning_venv\Scripts\python.exe`
- `EDGAR_LAPTOP`: `C:\Users\EdgarLangwald\OneDrive - neuland AI AG\Coding\.venv\Scripts\python.exe`
- `*.rwth-aachen.de` (RWTH cluster): `/rwthfs/rz/cluster/home/nld68820/.venv/bin/python`

Run all scripts and pip installs using the correct python for the current device.

## Other

EDGAR-PC and EDGAR_LAPTOP run on windows. Keep in mind that spaces in paths complicate the bash path finding.

