"""Embedding for LIMIT-v2 documents with per-model caching."""
import gc
import os

import numpy as np
import torch
from tqdm import tqdm

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
    """Download (if needed) and load the embedding model. Returns the model object."""
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


def _raw_embed(
    texts: list[str],
    model,
    model_name: str,
    is_query: bool,
    batch_size: int = 64,
) -> np.ndarray:
    """Return (len(texts), dim) float32 L2-normalized embeddings."""
    is_gritlm = MODELS[model_name].get("gritlm_api", False)
    prefix = MODELS[model_name].get("query_prefix" if is_query else "doc_prefix", "")
    suffix = MODELS[model_name].get("query_suffix", "") if is_query else ""

    if is_gritlm:
        embs = np.array(model.encode(texts, instruction=prefix, batch_size=batch_size), dtype=np.float32)
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


def embed_dataset(
    docs: list[dict],
    model_name: str,
    dataset_name: str,
    force: bool = False,
    batch_size: int = 64,
    device: str | None = None,
) -> np.ndarray:
    """Embed documents, cache to disk, and return (n, dim) float32 array.

    Each document is all of its sentences joined with a space.
    Cache layout: embeddings/{dataset_name}/{model_name}_d.npy
    """
    cache_path = os.path.join(_DEFAULT_CACHE_DIR, dataset_name, model_name + "_d.npy")
    doc_texts  = [" ".join(doc["sentences"]) for doc in docs]

    if not force and os.path.isfile(cache_path):
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape[0] == len(doc_texts):
            print(f"Loading {len(doc_texts)} cached embeddings ({dataset_name}/{model_name})")
            return np.array(cached)  # copy out of mmap for safety

    print(f"Embedding {len(doc_texts)} documents with {model_name} ...")
    model     = load_model(model_name, device)
    doc_prefix = MODELS[model_name].get("doc_prefix", "")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    batches = []
    for start in tqdm(range(0, len(doc_texts), batch_size), desc="docs"):
        batch  = doc_texts[start : start + batch_size]
        batches.append(_raw_embed(batch, model, model_name, is_query=False, batch_size=batch_size))

    doc_embs = np.concatenate(batches, axis=0)
    np.save(cache_path, doc_embs)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return doc_embs
