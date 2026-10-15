"""Unit tests for retrieval evaluation metrics; no API calls required."""

import pytest

from app.evaluation import hit_at_k, precision_at_k, reciprocal_rank, recall_at_k


def test_hit_at_k_finds_any_relevant_chunk():
    assert hit_at_k(["chunk_001", "chunk_003"], {"chunk_003"}) == 1
    assert hit_at_k(["chunk_001", "chunk_002"], {"chunk_003"}) == 0


def test_precision_at_k_uses_k_as_denominator():
    assert precision_at_k(["a", "b", "c", "d"], {"b", "d"}, 4) == pytest.approx(0.5)
    assert precision_at_k(["b"], {"b"}, 4) == pytest.approx(0.25)


def test_reciprocal_rank_uses_first_relevant_result():
    assert reciprocal_rank(["a", "b", "c"], {"b", "c"}) == pytest.approx(0.5)
    assert reciprocal_rank(["a", "b"], {"c"}) == 0.0


def test_precision_requires_positive_k():
    with pytest.raises(ValueError, match="k must be at least 1"):
        precision_at_k(["a"], {"a"}, 0)

def test_recall_at_k_measures_relevant_chunks_retrieved():
    assert recall_at_k(
        ["a", "b", "x", "y"],
        {"a", "b", "c", "d"},
        4,
    ) == pytest.approx(0.5)


def test_recall_at_k_uses_only_top_k_results():
    assert recall_at_k(
        ["x", "a", "b"],
        {"a", "b"},
        1,
    ) == pytest.approx(0.0)


def test_recall_at_k_returns_zero_with_no_relevant_ids():
    assert recall_at_k(["a", "b"], set(), 2) == 0.0


def test_recall_at_k_requires_positive_k():
    with pytest.raises(ValueError, match="k must be at least 1"):
        recall_at_k(["a"], {"a"}, 0)
