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
|
├── src/                          — importable package (the three pipeline stages)
├── dataset/
|
├── embeddings/ models/           — cached embeddings and model weights. !!These are symlinks to work partition!!
├── results/                      — evaluate() JSON outputs + report.pdf from visualize_results()
│   └── plot_studio/              — interactive/standalone result plotters (core.py data layer; main.py GUI)
|
├── replicate_official_results/   — per-model embedding implementations using each provider's official API

```

The tree HAS to be kept updated with PERMANENT folders. If you create a new one, make a judgement on wether the users intent is to keep the it in the long run, or whether its purpose is temporary (bugfix, exploration, etc.) 

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

### Query types

Every draw in `eval_targets.json` carries **5** query texts, all targeting the *same* 8 profiles
(`src.generate.QUERY_TYPES`, in this order — the order is also the index legend of evaluate()'s
per-type score lists):

| type            | source                | example |
|-----------------|-----------------------|---------|
| `broad`         | LLM (query_workflow)  | "Who plays a woodwind, …?" — each trait narrowed to a sub-family, names no item |
| `specific`      | LLM (query_workflow)  | precise descriptions, still names no item |
| `exact`         | LLM (exact_workflow)  | **names every item** + mirrors each template's focus: "Who counts cinnamon as their favorite scent, has played the bagpipe since childhood, and was born and raised in Orlando?" |
| `auto_sentence` | algorithmic           | "Which person has the following three attributes: scents: cinnamon, instruments: bagpipe, us_cities: Orlando?" |
| `auto_raw`      | algorithmic           | "scents: cinnamon, instruments: bagpipe, us_cities: Orlando" |

`generate_dataset` emits one query per type per draw with self-describing keys `"<draw_id>__<type>"`
and returns a per-query `query_types` list (aligned with `queries`/`qrels` by construction).
`evaluate(…, query_types)` groups by that list and writes each metric as a list over the types;
results are **never** pooled across types.

**Rebuilding the exact + auto queries** (broad/specific come from the separate query_workflow):
```bash
PYTHONPATH=. python dataset/build_exact_workflow_input.py --dump-specs   # 1. chunk specs (4 draws/chunk)
# 2. Workflow({ scriptPath: "dataset/exact_workflow.js" })               #    ~50 Haiku agents
PYTHONPATH=. python dataset/build_exact_workflow_input.py --render       # 3. merge query_exact + bake auto_*
PYTHONPATH=. python dataset/build_exact_workflow_input.py --auto-only    # (re)bake only auto_* (no LLM)
```
Adding/removing a query type means regenerating the `generated_datasets/` cache (it stores the
queries) and re-embedding queries. Doc embeddings are unaffected — the corpus is identical — so
`generate_dataset` reuses a pre-query_types corpus cache instead of regenerating fillers, and
`embed_dataset(..., only_embed="queries")` re-embeds only the query side, leaving the doc cache
untouched (`only_embed` is "docs" | "queries" | None and is independent of `force`).

## Coding practices

**IMPORTANT:** Use the correct venv based on the machine name. To get it, run `hostname` in bash:
- `EDGAR-PC`: `C:\Users\edgar\Projekte\Python\Machine_Learning_venv\Scripts\python.exe`
- `EDGAR_LAPTOP`: `C:\Users\EdgarLangwald\Documents\Coding\.venv\Scripts\python.exe` (shared across LIMIT-v2 and LIMIT)
- `*.rwth-aachen.de` (RWTH cluster): `/rwthfs/rz/cluster/home/nld68820/.venv/bin/python`

Run all scripts and pip installs using the correct python for the current device.

## Other

EDGAR-PC and EDGAR_LAPTOP run on windows. Keep in mind that spaces in paths complicate the bash path finding.

