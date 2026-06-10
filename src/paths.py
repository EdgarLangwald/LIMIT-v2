"""Canonical project paths, anchored to the repo root.

Import path constants from here instead of recomputing them in each module,
so generated data (``dataset/``, ``embeddings/``, ``models/``) always lands at
the repo root regardless of where the importing module lives.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR           = PROJECT_ROOT / "dataset"
GENERATED_DATASETS_DIR = DATASET_DIR / "generated_datasets"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
MODELS_DIR     = PROJECT_ROOT / "models"
ENV_PATH       = PROJECT_ROOT / ".env"

POOLS_DIR       = Path(__file__).resolve().parent / "pools"
EVAL_TARGETS    = POOLS_DIR / "eval_targets.json"
RESULTS_DIR     = PROJECT_ROOT / "results"
