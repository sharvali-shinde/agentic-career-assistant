"""Load career documents and create simple structure-aware chunks."""

import os
import re
from typing import Optional

from pypdf import PdfReader

try:
    from .config import CAREER_INFO_PDF_PATH, LINKEDIN_URL, SUMMARY_PATH
except ImportError:  # Support running this file directly as a script.
    from config import CAREER_INFO_PDF_PATH, LINKEDIN_URL, SUMMARY_PATH


class ProfileKnowledge:
    """Load a PDF and an optional text summary, then create chunks."""

    DATE_PATTERN = re.compile(
        r"(?i)(?:\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
        r"[a-z]*\.?\s+\d{4}\b|\b\d{4}\s*[-–—]\s*(?:\d{4}|present|current)\b)"
    )
    BULLET_PATTERN = re.compile(r"^\s*(?:[•●▪◦‣*]|o|\d+[.)])\s+")

    def __init__(self):
        self.documents = self._load_documents()
        self.chunks = self._chunk_documents()

    def _load_documents(self) -> list[dict[str, str]]:
        """Load the main PDF and include summary.txt only when it exists."""
        documents = []
        reader = PdfReader(CAREER_INFO_PDF_PATH)
        pages = []

        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())

        if pages:
            pdf_text = "\n\n".join(pages)
            if LINKEDIN_URL:
                pdf_text = f"Profile URL: {LINKEDIN_URL}\n\n{pdf_text}"
            documents.append({"source": "career_pdf", "text": pdf_text})

        # A missing summary is optional and should not stop the app from loading.
        if SUMMARY_PATH and os.path.isfile(SUMMARY_PATH):
            with open(SUMMARY_PATH, "r", encoding="utf-8") as file:
                summary = file.read().strip()
            if summary:
                documents.append({"source": "profile_summary", "text": summary})

        return documents

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace while retaining useful line boundaries."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def _is_bullet(cls, line: str) -> bool:
        return bool(cls.BULLET_PATTERN.match(line))

    @classmethod
    def _is_date(cls, line: str) -> bool:
        return bool(cls.DATE_PATTERN.search(line))

    @classmethod
    def _looks_like_heading(cls, line: str) -> bool:
        """Use generic line shape, not a list of known heading names."""
        value = line.strip().rstrip(":").strip()
        if not value or len(value) > 100 or cls._is_bullet(value):
            return False
        if len(value.split()) > 12 or value.endswith(('.', ',', ';', '?', '!')):
            return False
        if value[0].islower():
            return False
        # Sentence-like lines are more likely content than headings.
        if re.search(
            r"\b(is|are|was|were|built|developed|created|designed|worked|"
            r"led|used|have|has|will|can|with|for|from)\b",
            value,
            re.IGNORECASE,
        ):
            return False
        return True

    @classmethod
    def _is_section_heading(cls, line: str) -> bool:
        """Infer broad headings from formatting-like text cues."""
        if not cls._looks_like_heading(line):
            return False
        letters = [char for char in line if char.isalpha()]
        if not letters:
            return False
        uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
        return uppercase_ratio >= 0.8 or line.strip().endswith(":")

    @staticmethod
    def _slug(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "document"

    @classmethod
    def _build_semantic_units(cls, text: str, source: str) -> list[dict[str, str]]:
        """Group related lines using headings, bullets, dates and nearby context."""
        lines = [line.strip() for line in text.splitlines()]
        units: list[dict[str, str]] = []
        section = "document"
        title = "document"
        buffer: list[str] = []

        def save_buffer():
            nonlocal buffer
            content = "\n".join(buffer).strip()
            if content:
                units.append({
                    "source": source,
                    "section": section,
                    "title": title or section,
                    "text": content,
                })
            buffer = []

        def next_nonempty(index: int) -> str:
            for candidate in lines[index + 1:]:
                if candidate:
                    return candidate
            return ""

        for index, line in enumerate(lines):
            if not line:
                if buffer and buffer[-1] != "":
                    buffer.append("")
                continue

            if cls._is_section_heading(line):
                save_buffer()
                section = cls._slug(line.rstrip(":"))
                title = line.rstrip(":")
                continue

            following = next_nonempty(index)
            previous_line = next(
                (item.strip() for item in reversed(buffer) if item.strip()),
                "",
            )

            # If the previous line looks unfinished, this line may be a
            # continuation rather than a new heading (e.g. a wrapped phrase).
            previous_line_is_incomplete = (
                bool(previous_line)
                and not re.search(r"[.!?;:]$", previous_line)
            )

            is_record_heading = (
                cls._looks_like_heading(line)
                and (cls._is_bullet(following) or cls._is_date(following))
                and not previous_line_is_incomplete
            )

            if is_record_heading:
                # Keep a preceding standalone label with the record when it
                # looks like context (for example, an organization name).
                previous = [item for item in buffer if item.strip()]
                prefix = []
                if len(previous) == 1 and cls._looks_like_heading(previous[0]):
                    prefix = previous
                    buffer = []
                save_buffer()
                title = line
                buffer = prefix + [line]
                continue

            # A date-bearing line can start a new record when there is already
            # content in the buffer; otherwise it remains with its surrounding text.
            if cls._is_date(line) and buffer and any(item.strip() for item in buffer):
                save_buffer()
                title = line

            buffer.append(line)

        save_buffer()

        # Merge standalone, heading-shaped units into the following content
        # unit when they belong to the same source and section. This is generic:
        # it does not depend on particular employer, role, or project names.
        merged_units: list[dict[str, str]] = []
        pending_headings: list[dict[str, str]] = []

        def flush_pending() -> None:
            merged_units.extend(pending_headings)
            pending_headings.clear()

        for unit in units:
            unit_text = unit["text"].strip()
            is_standalone_heading = (
                "\n" not in unit_text
                and len(unit_text) <= 100
                and cls._looks_like_heading(unit_text)
            )

            if is_standalone_heading:
                if pending_headings and (
                    unit["source"] != pending_headings[0]["source"]
                    or unit["section"] != pending_headings[0]["section"]
                ):
                    flush_pending()
                pending_headings.append(unit)
                continue

            if pending_headings:
                can_merge = all(
                    heading["source"] == unit["source"]
                    and heading["section"] == unit["section"]
                    for heading in pending_headings
                )
                if can_merge:
                    unit = dict(unit)
                    prefix = "\n".join(
                        heading["text"].strip() for heading in pending_headings
                    )
                    unit["text"] = f"{prefix}\n{unit_text}"
                else:
                    flush_pending()
                pending_headings.clear()

            merged_units.append(unit)

        flush_pending()
        return merged_units

    @staticmethod
    def _split_long_unit(
        text: str, max_chars: int = 1200, overlap_chars: int = 150
    ) -> list[str]:
        """Split long text near a line or sentence boundary, with overlap."""
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if not 0 <= overlap_chars < max_chars:
            raise ValueError("overlap_chars must be non-negative and less than max_chars")

        text = text.strip()
        if not text:
            return []
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                search_start = start + max_chars // 2

                # Prefer a line break, then a sentence end, then a word boundary.
                line_boundary = text.rfind("\n", search_start, end)

                if line_boundary > start:
                    end = line_boundary
                else:
                    sentence_boundaries = [
                        text.rfind(". ", search_start, end),
                        text.rfind("? ", search_start, end),
                        text.rfind("! ", search_start, end),
                    ]
                    boundary = max(sentence_boundaries)

                    if boundary > start:
                        end = boundary + 1
                    else:
                        word_boundary = text.rfind(" ", search_start, end)
                        if word_boundary > start:
                            end = word_boundary

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break

            start = max(start + 1, end - overlap_chars)
            while start < end and not text[start].isspace():
                start += 1
            while start < end and text[start].isspace():
                start += 1

        return chunks

    def _chunk_documents(
        self, max_chars: int = 1200, overlap_chars: int = 150
    ) -> list[dict[str, str]]:
        """Create chunks with a common metadata shape across all sources."""
        chunks = []
        for document in self.documents:
            text = self._clean_text(document["text"])
            units = self._build_semantic_units(text, source=document["source"])
            for unit in units:
                for piece in self._split_long_unit(unit["text"], max_chars, overlap_chars):
                    chunks.append({
                        "source": unit["source"],
                        "section": unit["section"],
                        "title": unit["title"],
                        "text": piece,
                    })

        for index, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"chunk_{index:03d}"
        return chunks

    def get_context(self) -> str:
        """Keep the existing full-context interface for the agent."""
        return "\n\n".join(
            f"## Source: {document['source']}\n{document['text']}"
            for document in self.documents
        )


if __name__ == "__main__":
    knowledge = ProfileKnowledge()
    print(f"Documents loaded: {len(knowledge.documents)}")
    print(f"Chunks created: {len(knowledge.chunks)}")
    for chunk in knowledge.chunks:
        print("\n" + "=" * 70)
        print(f"ID: {chunk['chunk_id']} | Source: {chunk['source']}")
        print(f"Section: {chunk['section']} | Title: {chunk['title']}")
        print(f"Characters: {len(chunk['text'])}")
        print(chunk["text"])
