"""Unit tests for CareerRetriever; these tests do not call the OpenAI API."""

import pytest

from app.vector_store import CareerRetriever


class FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FakeVectorStore:
    def __init__(self, matches):
        self.matches = matches
        self.last_query = None
        self.last_k = None

    def similarity_search_with_score(self, query, k):
        self.last_query = query
        self.last_k = k
        return self.matches[:k]


def test_empty_query_returns_no_results():
    retriever = CareerRetriever(vector_store=FakeVectorStore([]))
    assert retriever.search("   ") == []


def test_search_requires_loaded_index():
    retriever = CareerRetriever()
    with pytest.raises(RuntimeError, match=r"Call build\(\) or load\(\)"):
        retriever.search("What did I build?")


def test_search_returns_text_metadata_and_distance():
    document = FakeDocument(
        "Built an agentic AI system with LangChain.",
        {
            "chunk_id": "chunk_002",
            "source": "career_pdf",
            "section": "work_experience",
            "title": "Project II",
        },
    )
    store = FakeVectorStore([(document, 0.24)])
    retriever = CareerRetriever(vector_store=store)

    results = retriever.search("What did I build with LangChain?", k=2)

    assert len(results) == 1
    assert results[0]["text"] == "Built an agentic AI system with LangChain."
    assert results[0]["chunk_id"] == "chunk_002"
    assert results[0]["source"] == "career_pdf"
    assert results[0]["distance"] == pytest.approx(0.24)
    assert store.last_query == "What did I build with LangChain?"
    assert store.last_k == 2


def test_k_must_be_positive():
    retriever = CareerRetriever(vector_store=FakeVectorStore([]))
    with pytest.raises(ValueError, match="k must be at least 1"):
        retriever.search("career", k=0)


def test_build_requires_chunks():
    retriever = CareerRetriever()
    with pytest.raises(ValueError, match="No chunks supplied"):
        retriever.build()
