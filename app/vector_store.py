"""Semantic retrieval for career-profile chunks using OpenAI embeddings and FAISS.

From the repository root:
    python app/vector_store.py
        Load the saved index and run a sample retrieval query.

    python app/vector_store.py --query "What did I build using LangChain?"
        Load the saved index and search a custom question.

    python app/vector_store.py --rebuild
        Recreate embeddings and overwrite the saved index.
"""

from pathlib import Path
from typing import Any
import argparse
import os
import sys


class CareerRetriever:
    """Build, save, load, and query a local FAISS index of career chunks."""

    def __init__(
        self,
        chunks: list[dict[str, str]] | None = None,
        index_path: str | Path = "artifacts/career_vector_store",
        embedding_model: str = "text-embedding-3-small",
        vector_store: Any = None,
    ):
        self.chunks = chunks or []
        self.index_path = Path(index_path)
        self.embedding_model = embedding_model
        # Optional injection makes retrieval behavior testable without API calls.
        self.vector_store = vector_store

    def _make_embeddings(self):
        """Create the embedding client only when needed."""
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Install the RAG dependencies listed in requirements.txt."
            ) from exc

        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY was not found. Check your project .env file."
            )

        return OpenAIEmbeddings(model=self.embedding_model)

    @staticmethod
    def _to_documents(chunks: list[dict[str, str]]):
        """Convert existing chunk dictionaries into LangChain Documents."""
        try:
            from langchain_core.documents import Document
        except ImportError as exc:
            raise ImportError(
                "Install the RAG dependencies listed in requirements.txt."
            ) from exc

        documents = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue

            metadata = {
                key: str(chunk.get(key, ""))
                for key in ("chunk_id", "source", "section", "title")
            }
            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def build(self, save: bool = True):
        """Create a fresh FAISS index from the supplied chunks."""
        if not self.chunks:
            raise ValueError("No chunks supplied. Pass ProfileKnowledge.chunks.")

        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise ImportError(
                "Install the RAG dependencies listed in requirements.txt."
            ) from exc

        documents = self._to_documents(self.chunks)
        if not documents:
            raise ValueError("The supplied chunks contain no searchable text.")

        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self._make_embeddings(),
        )

        if save:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(self.index_path))

        return self

    def load(self):
        """Load a previously saved index you created and trust."""
        if not (self.index_path / "index.faiss").is_file():
            raise FileNotFoundError(
                f"No saved FAISS index found at {self.index_path}."
            )

        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise ImportError(
                "Install the RAG dependencies listed in requirements.txt."
            ) from exc

        self.vector_store = FAISS.load_local(
            str(self.index_path),
            self._make_embeddings(),
            # Only load an index you created and trust; FAISS persistence includes
            # a pickle file for document metadata.
            allow_dangerous_deserialization=True,
        )
        return self

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """Return top-k chunks; lower FAISS distance generally means a closer match."""
        if not isinstance(query, str) or not query.strip():
            return []
        if k < 1:
            raise ValueError("k must be at least 1.")
        if self.vector_store is None:
            raise RuntimeError(
                "No vector index is loaded. Call build() or load() first."
            )

        matches = self.vector_store.similarity_search_with_score(query.strip(), k=k)
        results = []
        for document, distance in matches:
            result = {
                "text": document.page_content,
                "distance": float(distance),
            }
            result.update(document.metadata or {})
            results.append(result)

        return results


def _load_project_environment(project_root: Path) -> None:
    """Load .env before importing app configuration or creating embeddings."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ImportError(
            "python-dotenv is required. Ensure it is listed in requirements.txt."
        ) from exc

    load_dotenv(project_root / ".env")


def _build_retriever(project_root: Path) -> CareerRetriever:
    """Load the current career chunks and rebuild the saved index."""
    from app.retrieval import ProfileKnowledge

    knowledge = ProfileKnowledge()
    print(f"Loaded {len(knowledge.documents)} source documents.")
    print(f"Created {len(knowledge.chunks)} chunks.")

    retriever = CareerRetriever(
        chunks=knowledge.chunks,
        index_path=project_root / "artifacts" / "career_vector_store",
    )
    retriever.build(save=True)
    print(f"Embedding model: {retriever.embedding_model}")
    print(f"Vector index saved to: {retriever.index_path}")
    print("Embedding and indexing complete.")
    return retriever


def _print_results(query: str, results: list[dict[str, Any]]) -> None:
    print(f'\nQuery: "{query}"')
    if not results:
        print("No matching chunks found.")
        return

    for rank, result in enumerate(results, start=1):
        print("\n" + "-" * 72)
        print(
            f"Rank {rank} | {result.get('chunk_id', 'unknown id')} | "
            f"distance={result['distance']:.4f}"
        )
        print(
            f"Source: {result.get('source', 'unknown')} | "
            f"Section: {result.get('section', 'unknown')} | "
            f"Title: {result.get('title', 'unknown')}"
        )
        print(result["text"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load or rebuild the career-profile vector index and test retrieval."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recreate embeddings and overwrite the saved FAISS index.",
    )
    parser.add_argument(
        "--query",
        default="What agentic AI systems have I built?",
        help="Question to search for in the saved index.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=4,
        help="Number of matching chunks to return (default: 4).",
    )
    args = parser.parse_args()

    if args.k < 1:
        parser.error("-k must be at least 1")

    project_root = Path(__file__).resolve().parents[1]
    root_string = str(project_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    _load_project_environment(project_root)

    index_path = project_root / "artifacts" / "career_vector_store"
    retriever = CareerRetriever(index_path=index_path)

    if args.rebuild:
        print("Rebuilding the vector index; this will call the embeddings API.")
        retriever = _build_retriever(project_root)
    elif (index_path / "index.faiss").is_file():
        print(f"Loading saved vector index from: {index_path}")
        retriever.load()
        print("Saved index loaded; no document embeddings are being rebuilt.")
    else:
        print(f"No saved index found at: {index_path}")
        print("Build it once with: python app/vector_store.py --rebuild")
        raise SystemExit(1)

    _print_results(args.query, retriever.search(args.query, k=args.k))


if __name__ == "__main__":
    main()
