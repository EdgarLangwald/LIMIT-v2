"""
Verifies each replicate_* function against the score/shape annotations
that appear as comments in the provider's official example code.
Only models with explicit reference values in official_doc_code.txt get score tests.

Run all:  pytest replicate_official_results/test_replicate.py -v
One model: pytest replicate_official_results/test_replicate.py -v -k qwen3_0_6b
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Qwen3-0.6B
# Reference: tensor([[0.7646, 0.1414], [0.1355, 0.6000]])
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qwen3_0_6b_result():
    from replicate_official_results.replicate import replicate_qwen3_0_6b
    return replicate_qwen3_0_6b()


def test_qwen3_0_6b(qwen3_0_6b_result):
    r = qwen3_0_6b_result
    assert r["query_embeddings"].shape == (2, 1024), f"got {r['query_embeddings'].shape}"
    assert r["document_embeddings"].shape == (2, 1024), f"got {r['document_embeddings'].shape}"
    sim = np.round(np.array(r["similarity"].cpu()), 4)
    np.testing.assert_array_equal(sim, np.array([[0.7646, 0.1414], [0.1355, 0.6000]]))


# ---------------------------------------------------------------------------
# Qwen3-8B
# Reference: tensor([[0.7493, 0.0751], [0.0880, 0.6318]])
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qwen3_8b_result():
    from replicate_official_results.replicate import replicate_qwen3_8b
    return replicate_qwen3_8b()


def test_qwen3_8b(qwen3_8b_result):
    r = qwen3_8b_result
    assert r["query_embeddings"].shape == (2, 4096), f"got {r['query_embeddings'].shape}"
    assert r["document_embeddings"].shape == (2, 4096), f"got {r['document_embeddings'].shape}"
    sim = np.round(np.array(r["similarity"].cpu()), 4)
    np.testing.assert_array_equal(sim, np.array([[0.7493, 0.0751], [0.0880, 0.6318]]))


# ---------------------------------------------------------------------------
# Octen-0.6B  — no reference scores
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def octen_0_6b_result():
    from replicate_official_results.replicate import replicate_octen_0_6b
    return replicate_octen_0_6b()


def test_octen_0_6b(octen_0_6b_result):
    emb = octen_0_6b_result["embeddings"]
    assert emb.shape == (2, 1024), f"got {emb.shape}"


# ---------------------------------------------------------------------------
# Octen-4B  — no reference scores
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def octen_4b_result():
    from replicate_official_results.replicate import replicate_octen_4b
    return replicate_octen_4b()


def test_octen_4b(octen_4b_result):
    emb = octen_4b_result["embeddings"]
    assert emb.shape == (2, 2560), f"got {emb.shape}"


# ---------------------------------------------------------------------------
# Snowflake Arctic Embed L v2.0  — no reference scores or dims
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def snowflake_v2_result():
    from replicate_official_results.replicate import replicate_snowflake_v2
    return replicate_snowflake_v2()


def test_snowflake_v2(snowflake_v2_result):
    r = snowflake_v2_result
    assert r["query_embeddings"].shape == (2, 1024), f"got {r['query_embeddings'].shape}"
    assert r["document_embeddings"].shape == (2, 1024), f"got {r['document_embeddings'].shape}"


# ---------------------------------------------------------------------------
# F2LLM-1.7B
# Reference: tensor([[0.6735, 0.8418, 0.7513, 0.8602]])
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def f2llm_1_7b_result():
    from replicate_official_results.replicate import replicate_f2llm_1_7b
    return replicate_f2llm_1_7b()


def test_f2llm_1_7b(f2llm_1_7b_result):
    r = f2llm_1_7b_result
    assert r["query_embedding"].shape == (2048,), f"got {r['query_embedding'].shape}"
    assert r["document_embeddings"].shape == (4, 2048), f"got {r['document_embeddings'].shape}"
    sim = np.round(np.array(r["similarity"].cpu()).flatten(), 4)
    np.testing.assert_array_equal(sim, np.array([0.6735, 0.8418, 0.7513, 0.8602]))


# ---------------------------------------------------------------------------
# F2LLM-4B
# Reference: tensor([[0.6348, 0.8547, 0.7168, 0.8356]])
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def f2llm_4b_result():
    from replicate_official_results.replicate import replicate_f2llm_4b
    return replicate_f2llm_4b()


def test_f2llm_4b(f2llm_4b_result):
    r = f2llm_4b_result
    assert r["query_embedding"].shape == (2560,), f"got {r['query_embedding'].shape}"
    assert r["document_embeddings"].shape == (4, 2560), f"got {r['document_embeddings'].shape}"
    sim = np.round(np.array(r["similarity"].cpu()).flatten(), 4)
    np.testing.assert_array_equal(sim, np.array([0.6348, 0.8547, 0.7168, 0.8356]))


# ---------------------------------------------------------------------------
# Jina Embeddings v5 text-small  — no reference scores or dims
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def jina_v5_result():
    from replicate_official_results.replicate import replicate_jina_v5
    return replicate_jina_v5()


def test_jina_v5(jina_v5_result):
    r = jina_v5_result
    assert r["query_embeddings"].shape == (1, 1024), f"got {r['query_embeddings'].shape}"
    assert r["document_embeddings"].shape == (1, 1024), f"got {r['document_embeddings'].shape}"


# ---------------------------------------------------------------------------
# Voyage-4-nano  — no reference scores
# Reference shapes: (2048,) (4, 2048)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def voyage_4_nano_result():
    from replicate_official_results.replicate import replicate_voyage_4_nano
    return replicate_voyage_4_nano()


def test_voyage_4_nano(voyage_4_nano_result):
    r = voyage_4_nano_result
    assert r["query_embedding"].shape == (2048,), f"got {r['query_embedding'].shape}"
    assert r["document_embeddings"].shape == (4, 2048), f"got {r['document_embeddings'].shape}"


# ---------------------------------------------------------------------------
# E5-Mistral-7B-Instruct  — no reference scores
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def e5_mistral_result():
    from replicate_official_results.replicate import replicate_e5_mistral
    return replicate_e5_mistral()


def test_e5_mistral(e5_mistral_result):
    r = e5_mistral_result
    assert r["query_embeddings"].shape == (2, 4096), f"got {r['query_embeddings'].shape}"
    assert r["document_embeddings"].shape == (2, 4096), f"got {r['document_embeddings'].shape}"
