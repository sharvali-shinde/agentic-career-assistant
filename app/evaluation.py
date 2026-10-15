"""Reproducible evaluation for the saved career-profile semantic retriever.

Run from the repository root:
    python app/evaluation.py
    python app/evaluation.py --k 5

The CSV test cases contain manually curated relevant chunk IDs. Review those labels
whenever the source documents or chunking strategy changes.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> int:
    """Return 1 if at least one relevant chunk appears in the top-k results."""
    return int(any(chunk_id in relevant_ids for chunk_id in retrieved_ids))


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k slots occupied by relevant chunks."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    return sum(chunk_id in relevant_ids for chunk_id in retrieved_ids[:k]) / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of all relevant chunks retrieved in the top-k results."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    if not relevant_ids:
        return 0.0

    retrieved_relevant = set(retrieved_ids[:k]) & relevant_ids
    return len(retrieved_relevant) / len(relevant_ids)

def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant result; zero if none are relevant."""
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def load_retriever(project_root: Path):
    """Load .env and the already-built FAISS index; do not rebuild embeddings."""
    root_string = str(project_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ImportError(
            "python-dotenv is required. Ensure it is installed from requirements.txt."
        ) from exc

    load_dotenv(project_root / ".env")

    from app.vector_store import CareerRetriever

    index_path = project_root / "artifacts" / "career_vector_store"
    if not (index_path / "index.faiss").is_file():
        raise FileNotFoundError(
            f"No saved vector index found at {index_path}. "
            "Build it first with: python app/vector_store.py --rebuild"
        )

    retriever = CareerRetriever(index_path=index_path)
    retriever.load()
    return retriever


def evaluate(retriever, cases: pd.DataFrame, k: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run each test question and return per-question results and aggregate metrics."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    rows = []
    for _, case in cases.iterrows():
        question = str(case["question"]).strip()
        relevant_ids = {
            value.strip()
            for value in str(case["relevant_chunk_ids"]).split(";")
            if value.strip()
        }
        results = retriever.search(question, k=k)
        retrieved_ids = [str(result.get("chunk_id", "")) for result in results]

        rows.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "question": question,
                "relevant_chunk_ids": ";".join(sorted(relevant_ids)),
                "retrieved_chunk_ids": ";".join(retrieved_ids),
                "top1_chunk_id": retrieved_ids[0] if retrieved_ids else "",
                "hit_at_k": hit_at_k(retrieved_ids, relevant_ids),
                "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
                "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
                "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
                "top_results": " | ".join(
                    f"{result.get('chunk_id', '?')}: {result.get('title', '')}"
                    for result in results
                ),
            }
        )

    results_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(
        [
            {
                "questions": len(results_df),
                "k": k,
                "hit_rate_at_k": results_df["hit_at_k"].mean() if len(results_df) else 0.0,
                "mean_precision_at_k": results_df["precision_at_k"].mean() if len(results_df) else 0.0,
                "mean_recall_at_k": results_df["recall_at_k"].mean() if len(results_df) else 0.0,
                "mrr": results_df["reciprocal_rank"].mean() if len(results_df) else 0.0,

            }
        ]
    )
    return results_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate career-profile semantic retrieval.")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve per question.")
    parser.add_argument(
        "--cases",
        default="data/retrieval_eval_cases.csv",
        help="Path to the CSV test set, relative to the repository root by default.",
    )
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    project_root = Path(__file__).resolve().parents[1]
    cases_path = Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = project_root / cases_path
    if not cases_path.is_file():
        raise FileNotFoundError(f"Evaluation cases CSV not found: {cases_path}")

    cases = pd.read_csv(cases_path, dtype=str).fillna("")
    required_columns = {"case_id", "category", "question", "relevant_chunk_ids"}
    missing = required_columns - set(cases.columns)
    if missing:
        raise ValueError(f"Evaluation CSV is missing columns: {', '.join(sorted(missing))}")

    retriever = load_retriever(project_root)
    results_df, summary_df = evaluate(retriever, cases, k=args.k)

    output_dir = project_root / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "retrieval_eval_results.csv"
    summary_path = output_dir / "retrieval_eval_summary.csv"
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Loaded saved index and evaluated {len(cases)} questions (top-k={args.k}).")
    print("\nAggregate metrics:")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nPer-question results:")
    print(
        results_df[
            ["case_id", "category", "hit_at_k", "precision_at_k", "recall_at_k", "reciprocal_rank", "retrieved_chunk_ids"]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(f"\nDetailed results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
