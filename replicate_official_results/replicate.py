"""
Official embedding API replications for each target model.
Each replicate_* function is a faithful reproduction of the provider's documented
example code — no custom wrappers, no prefix hacks, just the official API.

Run a single function:
    PYTHONPATH=. python -c "from replicate_official_results.replicate import replicate_qwen3_0_6b; replicate_qwen3_0_6b()"
"""
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from src.paths import MODELS_DIR


def _load(local_key: str, hf_id: str, **kwargs) -> SentenceTransformer:
    """Load from models dir, downloading there first if not yet present."""
    local = MODELS_DIR / local_key
    if not local.is_dir():
        from huggingface_hub import snapshot_download
        print(f"  Downloading {local_key} ({hf_id}) ...")
        snapshot_download(repo_id=hf_id, local_dir=str(local))
        print(f"  {local_key} download complete.")
    else:
        print(f"  {local_key} (local)")
    return SentenceTransformer(str(local), **kwargs)


# ---------------------------------------------------------------------------
# Qwen3-Embedding  (0.6B and 8B share identical API)
# Source: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
# ---------------------------------------------------------------------------

def replicate_qwen3_0_6b() -> dict:
    model = _load(
        "Qwen0.6b", "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
    )

    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    with torch.no_grad():
        # prompt_name="query" activates the built-in retrieval query prompt
        query_embeddings = model.encode(queries, prompt_name="query")
        document_embeddings = model.encode(documents)

    similarity = model.similarity(query_embeddings, document_embeddings)
    print(f"[qwen3_0_6b] similarity:\n{similarity}")
    # Expected: tensor([[0.7646, 0.1414], [0.1355, 0.6000]])
    return {"query_embeddings": query_embeddings, "document_embeddings": document_embeddings, "similarity": similarity}


def replicate_qwen3_8b() -> dict:
    model = _load(
        "Qwen8b", "Qwen/Qwen3-Embedding-8B",
        model_kwargs={"torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
    )

    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    with torch.no_grad():
        query_embeddings = model.encode(queries, prompt_name="query")
        document_embeddings = model.encode(documents)

    similarity = model.similarity(query_embeddings, document_embeddings)
    print(f"[qwen3_8b] similarity:\n{similarity}")
    # Expected: tensor([[0.7493, 0.0751], [0.0880, 0.6318]])
    return {"query_embeddings": query_embeddings, "document_embeddings": document_embeddings, "similarity": similarity}


# ---------------------------------------------------------------------------
# Octen-Embedding  (0.6B and 4B share identical API)
# Source: https://huggingface.co/Octen/Octen-Embedding-0.6B
# NOTE: the official README shows only symmetric encode() with no query/doc split.
# ---------------------------------------------------------------------------

def replicate_octen_0_6b() -> dict:
    model = _load("Octen_0.6b", "Octen/Octen-Embedding-0.6B",
                  model_kwargs={"torch_dtype": torch.bfloat16})

    sentences = [
        "This is an example sentence",
        "Each sentence is converted to a vector",
    ]

    embeddings = model.encode(sentences)
    print(f"[octen_0_6b] shape: {embeddings.shape}")
    # Expected shape: (2, 1024)  — 0.6B has 1024-dim

    from sentence_transformers.util import cos_sim
    similarity = cos_sim(embeddings[0], embeddings[1])
    print(f"[octen_0_6b] similarity: {similarity.item():.4f}")
    return {"embeddings": embeddings, "similarity": similarity}


def replicate_octen_4b() -> dict:
    model = _load("Octen_4b", "Octen/Octen-Embedding-4B",
                  model_kwargs={"torch_dtype": torch.bfloat16})

    sentences = [
        "This is an example sentence",
        "Each sentence is converted to a vector",
    ]

    embeddings = model.encode(sentences)
    print(f"[octen_4b] shape: {embeddings.shape}")
    # Expected shape: (2, 2560)  — 4B has 2560-dim

    from sentence_transformers.util import cos_sim
    similarity = cos_sim(embeddings[0], embeddings[1])
    print(f"[octen_4b] similarity: {similarity.item():.4f}")
    return {"embeddings": embeddings, "similarity": similarity}


# ---------------------------------------------------------------------------
# Snowflake Arctic Embed L v2.0
# Source: https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0
# ---------------------------------------------------------------------------

def replicate_snowflake_v2() -> dict:
    model = _load("Snowflake_v2", "Snowflake/snowflake-arctic-embed-l-v2.0",
                  model_kwargs={"torch_dtype": torch.bfloat16})

    queries = ["what is snowflake?", "Where can I get the best tacos?"]
    documents = ["The Data Cloud!", "Mexico City of Course!"]

    query_embeddings = model.encode(queries, prompt_name="query")
    document_embeddings = model.encode(documents)

    scores = model.similarity(query_embeddings, document_embeddings)
    for query, query_scores in zip(queries, scores):
        doc_score_pairs = sorted(zip(documents, query_scores), key=lambda x: x[1], reverse=True)
        print(f"[snowflake_v2] Query: {query}")
        for doc, score in doc_score_pairs:
            print(f"  {score:.4f}  {doc}")
    return {"query_embeddings": query_embeddings, "document_embeddings": document_embeddings, "scores": scores}


# ---------------------------------------------------------------------------
# F2LLM  (1.7B and 4B share identical API via encode_query / encode_document)
# Source: https://huggingface.co/codefuse-ai/F2LLM-v2-1.7B
# ---------------------------------------------------------------------------

_F2LLM_QUERY = "What is F2LLM used for?"
_F2LLM_DOCS = [
    "We present F2LLM, a family of fully open embedding LLMs that achieve a strong balance between model size, training data, and embedding performance.",
    "F2LLM is a model for computing text embeddings that can be used for various NLP tasks such as information retrieval, semantic search, and text classification.",
    "F2LLM 是 CodeFuse 开源的系列嵌入模型。",
    "F2LLM — это модель вычисления встраивания текста, которую можно использовать для различных задач НЛП, таких как поиск информации, семантический поиск и классификация текста.",
]


def replicate_f2llm_1_7b() -> dict:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = _load(
        "F2LLM_1_7b", "codefuse-ai/F2LLM-v2-1.7B",
        device=device,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    query_embedding = model.encode_query(_F2LLM_QUERY)
    document_embeddings = model.encode_document(_F2LLM_DOCS)
    print(f"[f2llm_1_7b] shapes: {query_embedding.shape} {document_embeddings.shape}")
    # Expected: (2048,) (4, 2048)

    similarity = model.similarity(query_embedding, document_embeddings)
    print(f"[f2llm_1_7b] similarity: {similarity}")
    # Expected: tensor([[0.6735, 0.8418, 0.7513, 0.8602]])
    return {"query_embedding": query_embedding, "document_embeddings": document_embeddings, "similarity": similarity}


def replicate_f2llm_4b() -> dict:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = _load(
        "F2LLM_4b", "codefuse-ai/F2LLM-v2-4B",
        device=device,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    query_embedding = model.encode_query(_F2LLM_QUERY)
    document_embeddings = model.encode_document(_F2LLM_DOCS)
    print(f"[f2llm_4b] shapes: {query_embedding.shape} {document_embeddings.shape}")
    # Expected: (2560,) (4, 2560)

    similarity = model.similarity(query_embedding, document_embeddings)
    print(f"[f2llm_4b] similarity: {similarity}")
    # Expected: tensor([[0.6348, 0.8547, 0.7168, 0.8356]])
    return {"query_embedding": query_embedding, "document_embeddings": document_embeddings, "similarity": similarity}


# ---------------------------------------------------------------------------
# Jina Embeddings v5 text-small
# Source: https://huggingface.co/jinaai/jina-embeddings-v5-text-small
# ---------------------------------------------------------------------------

def replicate_jina_v5() -> dict:
    model = _load(
        "Jina_v5", "jinaai/jina-embeddings-v5-text-small",
        trust_remote_code=True,
        model_kwargs={"dtype": torch.bfloat16},
    )

    # task="retrieval" selects the retrieval head; prompt_name differentiates query vs document
    query_embeddings = model.encode(
        sentences=["Overview of climate change impacts on coastal cities"],
        task="retrieval",
        prompt_name="query",
    )
    document_embeddings = model.encode(
        sentences=["Climate change has led to rising sea levels, increased frequency of extreme weather events..."],
        task="retrieval",
        prompt_name="document",
    )
    print(f"[jina_v5] query shape:    {query_embeddings.shape}")
    print(f"[jina_v5] document shape: {document_embeddings.shape}")
    return {"query_embeddings": query_embeddings, "document_embeddings": document_embeddings}


# ---------------------------------------------------------------------------
# Voyage-4-nano
# Source: https://huggingface.co/voyageai/voyage-4-nano
# ---------------------------------------------------------------------------

def replicate_voyage_4_nano() -> dict:
    model = _load(
        "Voyage_4_nano", "voyageai/voyage-4-nano",
        trust_remote_code=True,
        truncate_dim=2048,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]

    query_embedding = model.encode_query(query)
    document_embeddings = model.encode_document(documents)
    print(f"[voyage_4_nano] query shape:    {query_embedding.shape}")
    # Expected: (2048,)
    print(f"[voyage_4_nano] document shape: {document_embeddings.shape}")
    # Expected: (4, 2048)
    return {"query_embedding": query_embedding, "document_embeddings": document_embeddings}


# ---------------------------------------------------------------------------
# E5-Mistral-7B-Instruct
# Source: https://huggingface.co/intfloat/e5-mistral-7b-instruct
# Both GritLM-7B and E5-Mistral-7B-Instruct are independent Mistral-7B fine-tunes;
# E5-Mistral is embedding-only and simpler to use.
# ---------------------------------------------------------------------------

def replicate_e5_mistral() -> dict:
    model = _load("E5_Mistral", "intfloat/e5-mistral-7b-instruct",
                  model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = 4096

    queries = [
        "how much protein should a female eat",
        "summit define",
    ]
    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]

    query_embeddings = model.encode(queries, prompt_name="web_search_query")
    document_embeddings = model.encode(documents)

    scores = (query_embeddings @ document_embeddings.T) * 100
    print(f"[e5_mistral] scores:\n{scores.tolist()}")
    return {"query_embeddings": query_embeddings, "document_embeddings": document_embeddings, "scores": scores}
