"""
Cross-validate AILAStatutes MTEB replication for Qwen3-Embedding-0.6B.

Three reference points:
  PUBLISHED — Qwen3 team (mteb 1.38.9, model revision b22da495)
  manual    — direct SentenceTransformer.encode() pipeline (same instruction & dataset revision)
  mteb_lib  — mteb.get_model() + mteb.evaluate() (installed mteb 2.x)

Run all:
    pytest replicate_official_results/test_AILAStatutes_qwen3.py -v
One approach only:
    pytest replicate_official_results/test_AILAStatutes_qwen3.py -v -k manual
    pytest replicate_official_results/test_AILAStatutes_qwen3.py -v -k mteb
"""
import pytest

from replicate_official_results.AILAStatutes_qwen3 import PUBLISHED

_METRICS = ["ndcg_at_10", "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "recall_at_20"]
ATOL = 0.0001


@pytest.fixture(scope="module")
def manual_result():
    from replicate_official_results.AILAStatutes_qwen3 import replicate_aila_manual
    return replicate_aila_manual()


@pytest.fixture(scope="module")
def mteb_result():
    from replicate_official_results.AILAStatutes_qwen3 import replicate_aila_mteb
    return replicate_aila_mteb()


# ── Manual vs Published ──────────────────────────────────────────────────────

class TestManualVsPublished:
    """Direct model.encode() pipeline vs. published Qwen3 MTEB scores."""

    def test_recall(self, manual_result):
        failures = [
            f"{m}: got {manual_result[m]:.5f}, expected: {PUBLISHED[m]:.5f}"
            for m in _METRICS
            if abs(manual_result[m] - PUBLISHED[m]) > ATOL
        ]
        if failures:
            pytest.fail("qwen3-embedding-0.6b (manual vs published)\n" + "\n".join(failures))


# ── MTEB lib vs Published ────────────────────────────────────────────────────

class TestMtebVsPublished:
    """mteb.evaluate() vs. published Qwen3 MTEB scores."""

    def test_recall(self, mteb_result):
        failures = [
            f"{m}: got {mteb_result[m]:.5f}, expected: {PUBLISHED[m]:.5f}"
            for m in _METRICS
            if mteb_result[m] is not None and abs(mteb_result[m] - PUBLISHED[m]) > ATOL
        ]
        if failures:
            pytest.fail("qwen3-embedding-0.6b (mteb vs published)\n" + "\n".join(failures))


# ── Manual vs MTEB lib ───────────────────────────────────────────────────────

class TestManualVsMteb:
    """Cross-check: both implementations should agree within tolerance."""

    def test_recall(self, manual_result, mteb_result):
        failures = [
            f"{m}: manual={manual_result[m]:.5f}, mteb={mteb_result[m]:.5f}, diff={abs(manual_result[m] - mteb_result[m]):.5f}"
            for m in _METRICS
            if mteb_result[m] is not None and abs(manual_result[m] - mteb_result[m]) > ATOL
        ]
        if failures:
            pytest.fail("qwen3-embedding-0.6b (manual vs mteb)\n" + "\n".join(failures))
