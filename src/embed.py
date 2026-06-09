"""Embedding for LIMIT-v2 with per-dataset disk cache and batch-level resume."""
import gc
import json
import os

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
import torch

from src.paths import EMBEDDINGS_DIR, MODELS_DIR

_DEFAULT_CACHE_DIR  = str(EMBEDDINGS_DIR)
_DEFAULT_MODELS_DIR = str(MODELS_DIR)
_device = "cuda" if torch.cuda.is_available() else "cpu"

MODELS: dict[str, dict] = {
    # ~0.1 GB; fast baseline, decent MTEB score
    "BGE_S": {
        "hf_id": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    # ~1.3 GB; strong MTEB ~2023, good general baseline
    "BGE_L": {
        "hf_id": "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    # ~0.5 GB; ModernBERT backbone, strong MTEB for its size (2024)
    "ModernBERT": {
        "hf_id": "Alibaba-NLP/gte-modernbert-base",
        "query_prefix": "",
    },
    # ~1.2 GB; SOTA-tier small model, strong MTEB for size (2025)
    "Qwen0.6b": {
        "hf_id": "Qwen/Qwen3-Embedding-0.6B",
        "query_prefix": "Instruct: Retrieve the person whose profile best answers the query.\nQuery:",
    },
    # ~8 GB; near SOTA on MTEB (2025)
    "Qwen4b": {
        "hf_id": "Qwen/Qwen3-Embedding-4B",
        "query_prefix": "Instruct: Retrieve the person whose profile best answers the query.\nQuery:",
    },
    # ~16 GB; SOTA on MTEB at release (2025)
    "Qwen8b": {
        "hf_id": "Qwen/Qwen3-Embedding-8B",
        "query_prefix": "Instruct: Retrieve the person whose profile best answers the query.\nQuery:",
    },
    # ~14 GB; was SOTA on MTEB at release (2023)
    "E5_Mistral": {
        "hf_id": "intfloat/e5-mistral-7b-instruct",
        "query_prefix": "Instruct: Retrieve the person whose profile best answers the query.\nQuery: ",
    },
    # ~14 GB; unified embedding+generation, strong MTEB (2024)
    "GritLM": {
        "hf_id": "GritLM/GritLM-7B",
        "query_prefix": "<|user|>\nRetrieve the person whose profile best answers the query.\n<|embed|>\n",
        "doc_prefix": "<|embed|>\n",
        "gritlm_api": True,
    },
    # ~16 GB; instruction-following retrieval based on Llama 3.1 (2024)
    "Promptriever": {
        "hf_id": "samaya-ai/promptriever-llama3.1-8b-instruct-v1",
        "query_prefix": "query:  ",
        "query_suffix": " Retrieve the person whose profile best answers the query.",
        "doc_prefix":   "passage:  ",
    },
    # ~16 GB; #1 MTEB late 2024, instruction-following with latent attention layer
    "NV_Embed": {
        "hf_id": "nvidia/NV-Embed-v2",
        "query_prefix": "Instruct: Retrieve the person whose profile best answers the query.\nQuery: ",
    },
    # ~14 GB; strong MTEB 2024, instruction-following, Qwen2 backbone
    "GTE_Qwen2": {
        "hf_id": "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "query_prefix": "Instruct: Retrieve the person whose profile best answers the query.\nQuery: ",
    },
    # ~1 GB; MoE architecture, 475M params (305M active), strong for size (2024)
    "Nomic_MoE": {
        "hf_id": "nomic-ai/nomic-embed-text-v2-moe",
        "query_prefix": "search_query: ",
    },
    # ~2 GB; updated Arctic-L, stronger MTEB than v1 (2024)
    "Snowflake_v2": {
        "hf_id": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "query_prefix": "query: ",
    },
}


def load_model(model_name: str, device: str | None = None):
    """Download (if needed) and load the embedding model."""
    model_id         = MODELS[model_name]["hf_id"]
    model_local_path = os.path.join(_DEFAULT_MODELS_DIR, model_name)
    use_device       = device or _device
    is_gritlm        = MODELS[model_name].get("gritlm_api", False)

    if not os.path.isdir(model_local_path):
        print(f"  Downloading {model_name} ({model_id}) ...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_dir=model_local_path)
    else:
        print(f"  {model_name} (local)")

    if is_gritlm:
        from transformers import MistralConfig
        from transformers.cache_utils import DynamicCache
        if not hasattr(MistralConfig, "rope_theta"):
            MistralConfig.rope_theta = 10000.0
        if not hasattr(DynamicCache, "from_legacy_cache"):
            DynamicCache.from_legacy_cache = classmethod(lambda cls, legacy=None: cls())
        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = lambda self, _, layer_idx=0: self.get_seq_length(layer_idx)
        from gritlm import GritLM as _GritLM
        return _GritLM(model_local_path, torch_dtype="auto", mode="embedding")
    else:
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_local_path, device=use_device)


def _raw_embed(texts: list[str], model, model_name: str, is_query: bool, batch_size: int = 64) -> np.ndarray:
    """Return (len(texts), dim) float32 L2-normalized embeddings."""
    is_gritlm = MODELS[model_name].get("gritlm_api", False)
    prefix    = MODELS[model_name].get("query_prefix" if is_query else "doc_prefix", "")
    suffix    = MODELS[model_name].get("query_suffix", "") if is_query else ""

    if is_gritlm:
        embs  = np.array(model.encode(texts, instruction=prefix, batch_size=batch_size), dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-8)
    else:
        processed = [prefix + t + suffix for t in texts] if (prefix or suffix) else texts
        return np.array(
            model.encode(processed, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


def embed_query(query: str, model_name: str, model) -> np.ndarray:
    """Embed a single query string. Returns (dim,) float32 array."""
    return _raw_embed([query], model, model_name, is_query=True, batch_size=1)[0]


def _progress_path(base: str) -> str:
    return base + "_progress.json"

def _load_progress(base: str) -> dict | None:
    p = _progress_path(base)
    return json.load(open(p)) if os.path.isfile(p) else None

def _save_progress(base: str, data: dict) -> None:
    with open(_progress_path(base), "w") as f:
        json.dump(data, f)

def _clear_progress(base: str) -> None:
    p = _progress_path(base)
    if os.path.isfile(p):
        os.remove(p)

def _cache_valid(base: str, n_doc: int, n_qry: int) -> bool:
    if os.path.isfile(_progress_path(base)):
        return False
    d_ok = (n_doc == 0) or (os.path.isfile(base + "_d.npy") and np.load(base + "_d.npy", mmap_mode="r").shape[0] == n_doc)
    q_ok = (n_qry == 0) or (os.path.isfile(base + "_q.npy") and np.load(base + "_q.npy", mmap_mode="r").shape[0] == n_qry)
    return d_ok and q_ok


def embed_dataset(
    dataset: dict,
    model_name: str,
    dataset_name: str,
    force: bool = False,
    batch_size: int = 64,
    device: str | None = None,
) -> dict:
    """Embed corpus and queries, cache to disk, return {"doc_embs": ndarray, "qry_embs": ndarray}.

    Skips embedding if corpus or queries dict is empty.
    Resumes interrupted runs batch-by-batch via progress files.
    Cache layout: embeddings/{dataset_name}/{model_name}_d.npy  /  _q.npy
    """
    doc_texts = list(dataset["corpus"].values())
    qry_texts = list(dataset["queries"].values())

    folder = os.path.join(_DEFAULT_CACHE_DIR, dataset_name)
    base   = os.path.join(folder, model_name)

    if not force and _cache_valid(base, len(doc_texts), len(qry_texts)):
        print(f"Loading cached embeddings ({dataset_name}/{model_name})")
        return _load_cached(base)

    os.makedirs(folder, exist_ok=True)
    model      = load_model(model_name, device)
    is_gritlm  = MODELS[model_name].get("gritlm_api", False)
    doc_prefix = MODELS[model_name].get("doc_prefix", "")

    if is_gritlm:
        dim = np.array(model.encode(["test"], instruction=doc_prefix, batch_size=1), dtype=np.float32).shape[-1]
    else:
        dim = model.get_embedding_dimension()

    progress   = None if force else _load_progress(base)
    docs_done  = (not force) and (progress is None or progress.get("docs_done", False)) and os.path.isfile(base + "_d.npy")
    if docs_done and doc_texts and np.load(base + "_d.npy", mmap_mode="r").shape[0] != len(doc_texts):
        docs_done = False
    doc_start  = progress.get("doc_start", 0) if (progress and not docs_done) else 0
    qry_start  = progress.get("qry_start", 0) if (progress and docs_done) else 0

    if doc_texts:
        doc_mm = open_memmap(base + "_d.npy", dtype="float32", mode="r+" if docs_done else "w+", shape=(len(doc_texts), dim))
        if not docs_done:
            n_batches = (len(doc_texts) + batch_size - 1) // batch_size
            for start in tqdm(range(doc_start, len(doc_texts), batch_size), desc="docs", initial=doc_start // batch_size, total=n_batches, mininterval=30):
                batch = doc_texts[start : start + batch_size]
                doc_mm[start : start + len(batch)] = _raw_embed(batch, model, model_name, is_query=False, batch_size=batch_size)
                _save_progress(base, {"docs_done": False, "doc_start": start + len(batch)})
            doc_mm.flush()
            _clear_progress(base)
        del doc_mm

    if qry_texts:
        qry_exists = os.path.isfile(base + "_q.npy")
        if qry_exists and np.load(base + "_q.npy", mmap_mode="r").shape[0] != len(qry_texts):
            qry_exists, qry_start = False, 0
        qry_mm = open_memmap(base + "_q.npy", dtype="float32", mode="r+" if qry_exists else "w+", shape=(len(qry_texts), dim))
        n_batches = (len(qry_texts) + batch_size - 1) // batch_size
        for start in tqdm(range(qry_start, len(qry_texts), batch_size), desc="queries", initial=qry_start // batch_size, total=n_batches, mininterval=30):
            batch = qry_texts[start : start + batch_size]
            qry_mm[start : start + len(batch)] = _raw_embed(batch, model, model_name, is_query=True, batch_size=batch_size)
            _save_progress(base, {"docs_done": True, "qry_start": start + len(batch)})
        qry_mm.flush()
        del qry_mm
        _clear_progress(base)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return _load_cached(base)


def _load_cached(base: str) -> dict:
    def _load(path):
        return np.load(path, mmap_mode="r") if os.path.isfile(path) else np.empty((0,), dtype=np.float32)
    return {"doc_embs": _load(base + "_d.npy"), "qry_embs": _load(base + "_q.npy")}
