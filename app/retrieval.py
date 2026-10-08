"""Career-profile retrieval and context preparation."""
from pypdf import PdfReader

from .config import (
    CAREER_INFO_PDF_PATH,
    LINKEDIN_URL,
    SUMMARY_PATH,
)


class ProfileKnowledge:

    def __init__(self):
        self.linkedin = self._load_linkedin()
        self.summary = self._load_summary()

    def _load_linkedin(self):
        reader = PdfReader(CAREER_INFO_PDF_PATH)

        linkedin = LINKEDIN_URL

        for page in reader.pages:
            text = page.extract_text()

            if text:
                linkedin += text

        return linkedin

    def _load_summary(self):
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def get_context(self):
        return (
            f"## Summary:\n{self.summary}\n\n"
            f"## LinkedIn Profile:\n{self.linkedin}"
        )