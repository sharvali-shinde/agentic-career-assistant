
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.vector_store import CareerRetriever


def tokenize(text: str) -> list[str]:
    """Normalize text into lowercase word tokens for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


class HybridCareerRetriever:
    """Combine semantic and BM25 rankings using Reciprocal Rank Fusion."""

    def __init__(
        self,
        index_path: str | Path = "artifacts/career_vector_store",
        embedding_model: str = "text-embedding-3-small",
        rrf_k: int = 60,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")

        self.rrf_k = rrf_k

        # Load the existing semantic index; do not rebuild it.
        self.semantic_retriever = CareerRetriever(
            index_path=Path(index_path),
            embedding_model=embedding_model,
        )
        self.semantic_retriever.load()

        vector_store = self.semantic_retriever.vector_store

        # Read documents in FAISS's internal document order.
        self.documents: list[dict[str, Any]] = []

        for docstore_id in vector_store.index_to_docstore_id.values():
            doc = vector_store.docstore.search(docstore_id)

            if not hasattr(doc, "page_content"):
                raise ValueError(
                    f"Could not load document for ID: {docstore_id}"
                )

            metadata = doc.metadata or {}
            record = {
                "text": doc.page_content,
                "chunk_id": metadata.get("chunk_id", docstore_id),
                "source": metadata.get("source"),
                "section": metadata.get("section"),
                "title": metadata.get("title"),
            }
            self.documents.append(record)

        if not self.documents:
            raise ValueError("The saved FAISS index contains no documents.")

        tokenized_corpus = [
            tokenize(document["text"]) for document in self.documents
        ]

        if any(not tokens for tokens in tokenized_corpus):
            raise ValueError("A document has no searchable word tokens.")

        self.bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _key(document: dict[str, Any]) -> str:
        """Return a stable identifier for ranking fusion."""
        chunk_id = document.get("chunk_id")
        if chunk_id is not None:
            return str(chunk_id)

        return (
            f"{document.get('source', '')}|"
            f"{document.get('title', '')}|"
            f"{document.get('text', '')}"
        )

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """Return the top-k documents ranked by semantic + BM25 RRF."""
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("Query contains no searchable word tokens.")

        corpus_size = len(self.documents)
        result_count = min(k, corpus_size)

        # Semantic rank: lower FAISS distance is better.
        semantic_results = self.semantic_retriever.search(
            query, k=corpus_size
        )

        # BM25 rank: higher BM25 score is better.
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_order = sorted(
            range(corpus_size),
            key=lambda i: (-float(bm25_scores[i]), i),
        )

        # RRF uses rank positions, not raw retrieval scores.
        fused_scores: dict[str, float] = defaultdict(float)
        records: dict[str, dict[str, Any]] = {}

        for rank, document in enumerate(semantic_results, start=1):
            key = self._key(document)
            records[key] = document
            fused_scores[key] += 1.0 / (self.rrf_k + rank)

        for rank, index in enumerate(bm25_order, start=1):
            document = self.documents[index]
            key = self._key(document)
            records[key] = document
            fused_scores[key] += 1.0 / (self.rrf_k + rank)

        ranked_keys = sorted(
            fused_scores,
            key=lambda key: (-fused_scores[key], key),
        )

        results = []
        for key in ranked_keys[:result_count]:
            result = dict(records[key])
            result["rrf_score"] = fused_scores[key]
            results.append(result)

        return results
