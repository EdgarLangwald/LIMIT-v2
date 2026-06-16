"""Embedding for LIMIT-v2 with per-dataset disk cache and batch-level resume."""
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
import torch

from src.paths import EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR

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
    # ~2.3 GB; multilingual XLM-RoBERTa-large; dense+sparse+ColBERT (SentenceTransformer gives dense).
    # Unlike bge-*-en-v1.5, M3 needs NO query instruction; 1024-dim, 8192 ctx (2024)
    "BGE_M3": {
        "hf_id": "BAAI/bge-m3",
        "query_prefix": "",
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
    # ~1 GB; MoE, 475M params (305M active), strong for size; 768-dim Matryoshka.
    # Asymmetric: documents MUST use the search_document prefix. Needs trust_remote_code (2024)
    "Nomic_Embed_v2": {
        "hf_id": "nomic-ai/nomic-embed-text-v2-moe",
        "query_prefix": "search_query: ",
        "doc_prefix":   "search_document: ",
        "trust_remote_code": True,
    },
    # ~2.3 GB; multilingual XLM-RoBERTa w/ task-specific LoRA adapters, 1024-dim Matryoshka, 8192 ctx.
    # Selects adapter + instruction via task/prompt_name encode args (NOT text prefixes); needs
    # trust_remote_code + `pip install einops`. License: CC-BY-NC-4.0 (non-commercial) (2024)
    "Jina_v3": {
        "hf_id": "jinaai/jina-embeddings-v3",
        "query_prefix": "",
        "doc_prefix":   "",
        "trust_remote_code": True,
        "query_encode_kwargs": {"task": "retrieval.query",   "prompt_name": "retrieval.query"},
        "doc_encode_kwargs":   {"task": "retrieval.passage", "prompt_name": "retrieval.passage"},
    },
    # ~2 GB; updated Arctic-L, stronger MTEB than v1 (2024)
    "Snowflake_v2": {
        "hf_id": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "query_prefix": "query: ",
    },
}


class _GritLMEmbedder:
    """Embedding-only replacement for the `gritlm` package.

    GritLM-7B ships custom remote modeling code written for transformers 4.37 that
    crashes on transformers 5.x, and the `gritlm` pip package depends on it. The
    weights are plain Mistral-7B though, so we load them into the stock
    `MistralModel` and use the transformers>=5 `config.is_causal = False` switch
    to get the bidirectional attention GritLM's embedding mode was trained with.
    Pooling replicates gritlm.GritLM.encode: mean over non-instruction tokens,
    right padding, max_length 512, L2 normalize.
    """

    def __init__(self, model_path: str, device: str):
        from transformers import AutoTokenizer, MistralModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        self.model = MistralModel.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa")
        self.model.config.is_causal = False      # bidirectional attention (the 'bb' in GritLM's bbcc)
        self.model.config.sliding_window = None  # full attention, as in the original embedding path
        self.model.eval().to(device)
        self.device = device
        print(f"  device: {next(self.model.parameters()).device}, dtype: {self.model.dtype}")

    @torch.no_grad()
    def encode(self, texts: list[str], instruction: str = "", batch_size: int = 64, max_length: int = 512) -> np.ndarray:
        # Instruction token count includes BOS, exactly like gritlm's masking
        n_inst = len(self.tokenizer(instruction)["input_ids"]) if instruction else 0
        out = []
        for i in range(0, len(texts), batch_size):
            batch  = [instruction + t for t in texts[i : i + batch_size]]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                                    return_tensors="pt").to(self.device)
            hidden = self.model(**inputs, use_cache=False).last_hidden_state
            mask   = inputs["attention_mask"].clone()
            mask[:, :n_inst] = 0  # mean-pool only over the text, not the instruction
            pooled = (hidden * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
            out.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu().numpy())
        return np.concatenate(out, axis=0)


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
        print(f"  {model_name} download complete.")
    else:
        print(f"  {model_name} (local)")

    if is_gritlm:
        return _GritLMEmbedder(model_local_path, device=use_device)
    else:
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        from sentence_transformers import SentenceTransformer
        st_kwargs = {"trust_remote_code": True} if MODELS[model_name].get("trust_remote_code") else {}
        if st_kwargs:
            _patch_remote_code_tied_weights()
        # Force fp32 on CPU: models with a bf16/fp16 config (e.g. Jina v3) produce
        # NaN embeddings under CPU attention without flash_attn. bf16 stays on GPU.
        if use_device == "cpu":
            st_kwargs["model_kwargs"] = {"dtype": torch.float32}
        st = SentenceTransformer(model_local_path, device=use_device, **st_kwargs)
        if st_kwargs.get("trust_remote_code"):
            _reinit_remote_code_buffers(st)
        print(f"  device: {next(st.parameters()).device}")
        return st


def _patch_remote_code_tied_weights() -> None:
    """Make pre-5.x trust_remote_code wrappers loadable under transformers 5.x.

    transformers 5.x reads `self.all_tied_weights_keys` directly in
    `_finalize_model_loading` (modeling_utils.py), an attribute that only the
    modern `post_init()` creates. Old custom wrappers (e.g. Jina v3's
    `XLMRobertaLoRA`) call `super().__init__(config)` without `post_init()`, so
    the top-level model never gets it and loading crashes with
    `AttributeError: ... has no attribute 'all_tied_weights_keys'`.

    These wrappers have no tied weights, and transformers itself treats the
    absent value as `{}` elsewhere, so we install `{}` as a class-level fallback.
    Properly initialized models override it per-instance in post_init(), so this
    is a no-op for them. Idempotent."""
    from transformers import PreTrainedModel
    if "all_tied_weights_keys" not in PreTrainedModel.__dict__:
        PreTrainedModel.all_tied_weights_keys = {}


def _reinit_remote_code_buffers(st) -> None:
    """Re-initialize non-persistent buffers that transformers 5.x leaves uninitialized.

    transformers 5.x instantiates models on the meta device, then materializes every
    non-persistent buffer with `torch.empty_like` (uninitialized memory) and relies on
    `_init_weights()` to fix them up. Old remote-code models (e.g. Jina v3) don't
    re-init their custom buffers there, so they keep garbage — which produces NaN /
    non-deterministic embeddings. Two buffers matter for Jina v3:

      * rotary `inv_freq` — used every forward. Its lazy-recompute only triggers when
        dtype != fp32 (rotary.py), so the fp32 load path reads the garbage buffer.
      * LoRA `lora_dropout_mask` — should be all ones (only read when dropout_p > 0).

    We recompute them from the modules' own init logic. No-op for models without them."""
    for module in st.modules():
        if hasattr(module, "_compute_inv_freq") and getattr(module, "inv_freq", None) is not None:
            module.inv_freq = module._compute_inv_freq(device=module.inv_freq.device)
            if hasattr(module, "_seq_len_cached"):
                module._seq_len_cached = 0
            for c in ("_cos_cached", "_sin_cached", "_cos_k_cached", "_sin_k_cached"):
                if hasattr(module, c):
                    setattr(module, c, None)
        if getattr(module, "lora_dropout_mask", None) is not None:
            module.lora_dropout_mask = torch.ones_like(module.lora_dropout_mask)


def _raw_embed(texts: list[str], model, model_name: str, is_query: bool, batch_size: int = 64) -> np.ndarray:
    """Return (len(texts), dim) float32 L2-normalized embeddings."""
    is_gritlm  = MODELS[model_name].get("gritlm_api", False)
    prefix     = MODELS[model_name].get("query_prefix" if is_query else "doc_prefix", "")
    suffix     = MODELS[model_name].get("query_suffix", "") if is_query else ""
    enc_kwargs = MODELS[model_name].get("query_encode_kwargs" if is_query else "doc_encode_kwargs", {})

    if is_gritlm:
        embs  = np.array(model.encode(texts, instruction=prefix, batch_size=batch_size), dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-8)
    else:
        processed = [prefix + t + suffix for t in texts] if (prefix or suffix) else texts
        return np.array(
            model.encode(processed, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False, **enc_kwargs),
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

def _array_ok(path: str, n: int) -> bool:
    """File exists, has n rows, and its boundary rows are non-zero (cheap corruption check:
    catches the zero-prefix/suffix left by a botched resume without scanning the whole file)."""
    if not os.path.isfile(path):
        return False
    arr = np.load(path, mmap_mode="r")
    if arr.shape[0] != n:
        return False
    return bool(np.linalg.norm(arr[0]) > 1e-6 and np.linalg.norm(arr[-1]) > 1e-6)

def _cache_valid(base: str, n_doc: int, n_qry: int) -> bool:
    if os.path.isfile(_progress_path(base)):
        return False
    d_ok = (n_doc == 0) or _array_ok(base + "_d.npy", n_doc)
    q_ok = (n_qry == 0) or _array_ok(base + "_q.npy", n_qry)
    return d_ok and q_ok


def _write_duration(model_name: str, dataset_stem: str, phase: str, avg_s: float, n_batches: int) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "durations.txt", "a") as f:
        f.write(f"{model_name}\t{dataset_stem}\t{phase}\t{avg_s:.4f}s/batch\t({n_batches} batches)\n")


def embed_dataset(
    dataset: dict,
    model_name: str,
    dataset_path: Path,
    force: bool = False,
    batch_size: int = 64,
    device: str | None = None,
    model=None,
    only_embed: str | None = None,
) -> tuple[dict, Path]:
    """Embed corpus and queries, cache to disk, return ({"doc_embs": ndarray, "qry_embs": ndarray}, embs_path).

    embs_path is EMBEDDINGS_DIR/<dataset_stem>/<model_name> — pass it to evaluate().
    Skips embedding if corpus or queries dict is empty.
    Resumes interrupted runs batch-by-batch via progress files.
    Pass an already-loaded *model* to skip the load_model() call.

    only_embed restricts which side is touched, independently of *force*:
      - None        : embed/refresh both docs and queries (default).
      - "docs"      : only the doc-embedding path runs; queries are left untouched.
      - "queries"   : only the query-embedding path runs; docs are left untouched.
    The skipped side's existing cache file (if any) is still returned via _load_cached.
    Use "queries" to re-embed queries after changing the query set without re-touching the
    (identical) corpus.
    """
    if only_embed not in (None, "docs", "queries"):
        raise ValueError(f"only_embed must be None, 'docs', or 'queries' — got {only_embed!r}")
    do_docs = only_embed != "queries"
    do_qry  = only_embed != "docs"

    doc_texts = list(dataset["corpus"].values())
    qry_texts = list(dataset["queries"].values())

    folder = os.path.join(_DEFAULT_CACHE_DIR, Path(dataset_path).stem)
    base   = os.path.join(folder, model_name)

    # Only require the side(s) we're being asked to embed to be cache-valid (passing 0 makes
    # _cache_valid treat the skipped side as already satisfied).
    if not force and _cache_valid(base, len(doc_texts) if do_docs else 0,
                                  len(qry_texts) if do_qry else 0):
        print(f"Loading cached embeddings ({Path(dataset_path).stem}/{model_name})")
        return _load_cached(base), Path(base)

    os.makedirs(folder, exist_ok=True)
    model      = model if model is not None else load_model(model_name, device)
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

    if doc_texts and do_docs:
        # Resume in-place ('r+') only if a correctly-shaped file already exists; opening 'w+'
        # would zero-fill a fresh file and silently discard batches written before an interrupt.
        d_path     = base + "_d.npy"
        can_resume = os.path.isfile(d_path) and tuple(np.load(d_path, mmap_mode="r").shape) == (len(doc_texts), dim)
        if not can_resume:
            doc_start = 0
        doc_mm = open_memmap(d_path, dtype="float32", mode="r+" if can_resume else "w+", shape=(len(doc_texts), dim))
        if not docs_done:
            _dur_total, _dur_count = 0.0, 0
            _last_print, _print_interval = time.monotonic(), 0.0
            for start in range(doc_start, len(doc_texts), batch_size):
                batch = doc_texts[start : start + batch_size]
                _t0 = time.monotonic()
                doc_mm[start : start + len(batch)] = _raw_embed(batch, model, model_name, is_query=False, batch_size=batch_size)
                _dur_total += time.monotonic() - _t0
                _dur_count += 1
                _save_progress(base, {"docs_done": False, "doc_start": start + len(batch)})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                now = time.monotonic()
                if now - _last_print >= _print_interval:
                    done = min(start + batch_size, len(doc_texts))
                    print(f"  docs {done:,}/{len(doc_texts):,} ({100*done/len(doc_texts):.1f}%)", flush=True)
                    _last_print = now
                    _print_interval = min(300.0, max(1.0, _print_interval * 2.0))
            doc_mm.flush()
            _clear_progress(base)
            _write_duration(model_name, Path(dataset_path).stem, "docs", _dur_total / _dur_count, _dur_count)
        del doc_mm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if qry_texts and do_qry:
        qry_exists = os.path.isfile(base + "_q.npy")
        if qry_exists and np.load(base + "_q.npy", mmap_mode="r").shape[0] != len(qry_texts):
            qry_exists, qry_start = False, 0
        qry_mm = open_memmap(base + "_q.npy", dtype="float32", mode="r+" if qry_exists else "w+", shape=(len(qry_texts), dim))
        _dur_total, _dur_count = 0.0, 0
        _last_print, _print_interval = time.monotonic(), 0.0
        for start in range(qry_start, len(qry_texts), batch_size):
            batch = qry_texts[start : start + batch_size]
            _t0 = time.monotonic()
            qry_mm[start : start + len(batch)] = _raw_embed(batch, model, model_name, is_query=True, batch_size=batch_size)
            _dur_total += time.monotonic() - _t0
            _dur_count += 1
            _save_progress(base, {"docs_done": True, "qry_start": start + len(batch)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            now = time.monotonic()
            if now - _last_print >= _print_interval:
                done = min(start + batch_size, len(qry_texts))
                print(f"  queries {done:,}/{len(qry_texts):,} ({100*done/len(qry_texts):.1f}%)", flush=True)
                _last_print = now
                _print_interval = min(300.0, max(1.0, _print_interval * 2.0))
        qry_mm.flush()
        del qry_mm
        _clear_progress(base)
        _write_duration(model_name, Path(dataset_path).stem, "queries", _dur_total / _dur_count, _dur_count)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Embedding complete.")

    return _load_cached(base), Path(base)


def _load_cached(base: str) -> dict:
    def _load(path):
        return np.load(path, mmap_mode="r") if os.path.isfile(path) else np.empty((0,), dtype=np.float32)
    return {"doc_embs": _load(base + "_d.npy"), "qry_embs": _load(base + "_q.npy")}
