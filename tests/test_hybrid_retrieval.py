
from app.hybrid_retrieval import HybridCareerRetriever, tokenize


def test_tokenize_lowercases_and_splits_text():
    assert tokenize("Loan Payback Predictor!") == [
        "loan",
        "payback",
        "predictor",
    ]


def test_rrf_rewards_documents_ranked_by_both_methods():
    retriever = HybridCareerRetriever.__new__(HybridCareerRetriever)
    retriever.rrf_k = 60
    retriever.documents = [
        {"chunk_id": "A", "text": "alpha"},
        {"chunk_id": "B", "text": "beta"},
        {"chunk_id": "C", "text": "gamma"},
    ]

    # Avoid constructing the FAISS index or calling OpenAI.
    class FakeSemanticRetriever:
        def search(self, query, k):
            return [
                {"chunk_id": "A", "text": "alpha", "distance": 0.1},
                {"chunk_id": "B", "text": "beta", "distance": 0.2},
                {"chunk_id": "C", "text": "gamma", "distance": 0.3},
            ]

    class FakeBM25:
        def get_scores(self, tokens):
            # BM25 order: B, A, C
            return [2.0, 3.0, 1.0]

    retriever.semantic_retriever = FakeSemanticRetriever()
    retriever.bm25 = FakeBM25()

    results = retriever.search("alpha beta", k=3)

    assert [item["chunk_id"] for item in results] == ["A", "B", "C"]
    assert results[0]["rrf_score"] == 1 / 61 + 1 / 62
    assert results[1]["rrf_score"] == 1 / 62 + 1 / 61
    assert results[2]["rrf_score"] == 1 / 63 + 1 / 63


def test_search_rejects_empty_query():
    retriever = HybridCareerRetriever.__new__(HybridCareerRetriever)

    try:
        retriever.search("   ", k=3)
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for an empty query")


def test_search_rejects_invalid_k():
    retriever = HybridCareerRetriever.__new__(HybridCareerRetriever)

    try:
        retriever.search("test", k=0)
    except ValueError as exc:
        assert "at least 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for k=0")
