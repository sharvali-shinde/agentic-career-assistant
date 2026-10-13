"""Tests for generic structure-aware chunking in app.retrieval."""

import os
import tempfile
from unittest.mock import patch

import pytest

import app.retrieval as retrieval_module
from app.retrieval import ProfileKnowledge


def test_short_text_remains_one_chunk():
    assert ProfileKnowledge._split_long_unit("Hello world.", 1200, 150) == [
        "Hello world."
    ]


def test_long_text_is_split_within_maximum_size():
    long_text = " ".join(["career"] * 1000)
    chunks = ProfileKnowledge._split_long_unit(
        long_text, max_chars=1200, overlap_chars=150
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 1200 for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_empty_text_returns_no_chunks():
    assert ProfileKnowledge._split_long_unit("", 1200, 150) == []
    assert ProfileKnowledge._split_long_unit("   ", 1200, 150) == []


@pytest.mark.parametrize(
    ("text", "max_chars", "overlap_chars"),
    [
        ("test", 0, 0),
        ("test", 100, -1),
        ("test", 100, 100),
    ],
)
def test_invalid_split_parameters_raise_value_error(
    text, max_chars, overlap_chars
):
    with pytest.raises(ValueError):
        ProfileKnowledge._split_long_unit(text, max_chars, overlap_chars)


def test_parser_uses_generic_structure_not_known_names():
    sample = """SUMMARY
I build reliable machine learning systems.

WORK EXPERIENCE
Northstar Analytics
Machine Learning Engineer Jan 2024 - present
• Built a customer churn forecasting model.
• Improved model monitoring.

PROJECTS AND LEADERSHIP
Customer Churn Forecasting
• Built a forecasting pipeline using Python.

Inventory Risk Dashboard
• Created a dashboard for inventory risk.
"""
    units = ProfileKnowledge._build_semantic_units(
        sample, source="synthetic_test"
    )

    assert any(
        unit["section"] == "summary"
        and "reliable machine learning systems" in unit["text"]
        for unit in units
    )
    assert any(
        unit["section"] == "work_experience"
        and "customer churn forecasting model" in unit["text"]
        for unit in units
    )
    assert any(
        unit["section"] == "projects_and_leadership"
        and "Customer Churn Forecasting" in unit["text"]
        and "forecasting pipeline" in unit["text"]
        for unit in units
    )
    assert any(
        unit["section"] == "projects_and_leadership"
        and "Inventory Risk Dashboard" in unit["text"]
        and "dashboard for inventory risk" in unit["text"]
        for unit in units
    )


class FakePage:
    def extract_text(self):
        return "TECHNICAL NOTES\nBuilt a reusable search pipeline."


class FakeReader:
    def __init__(self, _path):
        self.pages = [FakePage()]


def test_missing_summary_is_optional():
    with tempfile.TemporaryDirectory() as temp_dir:
        missing_summary = os.path.join(temp_dir, "summary.txt")

        with (
            patch.object(retrieval_module, "CAREER_INFO_PDF_PATH", "fake.pdf"),
            patch.object(retrieval_module, "LINKEDIN_URL", ""),
            patch.object(retrieval_module, "SUMMARY_PATH", missing_summary),
            patch.object(retrieval_module, "PdfReader", FakeReader),
        ):
            knowledge = ProfileKnowledge()

        assert len(knowledge.documents) == 1
        assert knowledge.documents[0]["source"] == "career_pdf"


def test_existing_summary_is_loaded_as_separate_source():
    with tempfile.TemporaryDirectory() as temp_dir:
        summary_path = os.path.join(temp_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as file:
            file.write("A concise professional summary.")

        with (
            patch.object(retrieval_module, "CAREER_INFO_PDF_PATH", "fake.pdf"),
            patch.object(retrieval_module, "LINKEDIN_URL", ""),
            patch.object(retrieval_module, "SUMMARY_PATH", summary_path),
            patch.object(retrieval_module, "PdfReader", FakeReader),
        ):
            knowledge = ProfileKnowledge()

        assert [doc["source"] for doc in knowledge.documents] == [
            "career_pdf",
            "profile_summary",
        ]
        assert knowledge.documents[1]["text"] == (
            "A concise professional summary."
        )


def test_standalone_heading_merges_into_following_content_generically():
    sample = """WORK EXPERIENCE
Northstar Analytics
Jan 2024 - present
• Built a customer churn forecasting model.
• Improved model monitoring.
"""
    units = ProfileKnowledge._build_semantic_units(
        sample, source="synthetic_test"
    )

    assert not any(unit["text"].strip() == "Northstar Analytics" for unit in units)
    assert any(
        "Northstar Analytics" in unit["text"]
        and "customer churn forecasting model" in unit["text"]
        for unit in units
    )
