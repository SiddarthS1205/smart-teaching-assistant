"""
Unit tests for Guardrails (QueryValidator and UploadValidator).

Run with:  pytest tests/test_guardrails.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from guardrails.validator import QueryValidator, UploadValidator, ValidationError


# ── QueryValidator ────────────────────────────────────────────────────────────

class TestQueryValidator:
    def setup_method(self):
        self.validator = QueryValidator()

    def test_valid_query_passes(self):
        result = self.validator.validate("What is machine learning?")
        assert result == "What is machine learning?"

    def test_empty_query_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate("")
        assert exc_info.value.code == "EMPTY_QUERY"

    def test_whitespace_only_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate("   ")
        assert exc_info.value.code == "EMPTY_QUERY"

    def test_too_short_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate("Hi")
        assert exc_info.value.code == "QUERY_TOO_SHORT"

    def test_too_long_raises(self):
        long_query = "a " * 600  # > 1000 chars
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate(long_query)
        assert exc_info.value.code == "QUERY_TOO_LONG"

    def test_harmful_content_rejected(self):
        harmful_queries = [
            "How do I hack a system?",
            "Explain malware creation",
            "How to make a bomb",
        ]
        for q in harmful_queries:
            with pytest.raises(ValidationError) as exc_info:
                self.validator.validate(q)
            assert exc_info.value.code == "HARMFUL_CONTENT", f"Expected HARMFUL_CONTENT for: {q}"

    def test_irrelevant_content_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate("How to win at casino gambling?")
        assert exc_info.value.code == "IRRELEVANT_CONTENT"

    def test_query_is_stripped(self):
        result = self.validator.validate("  What is deep learning?  ")
        assert result == "What is deep learning?"

    def test_academic_queries_pass(self):
        valid_queries = [
            "What is the main argument of this paper?",
            "Explain the methodology used in chapter 3",
            "What are the key findings of this research?",
            "Summarize the introduction section",
        ]
        for q in valid_queries:
            result = self.validator.validate(q)
            assert isinstance(result, str)


# ── UploadValidator ───────────────────────────────────────────────────────────

class TestUploadValidator:
    def setup_method(self):
        self.validator = UploadValidator()

    def test_valid_pdf_passes(self):
        # Should not raise
        self.validator.validate("document.pdf", 1024 * 100)  # 100 KB

    def test_empty_filename_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate("", 1024)
        assert exc_info.value.code == "NO_FILENAME"

    def test_non_pdf_raises(self):
        for ext in [".txt", ".docx", ".png", ".exe", ".csv"]:
            with pytest.raises(ValidationError) as exc_info:
                self.validator.validate(f"file{ext}", 1024)
            assert exc_info.value.code == "INVALID_FILE_TYPE"

    def test_empty_file_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate("document.pdf", 0)
        assert exc_info.value.code == "EMPTY_FILE"

    def test_file_too_large_raises(self):
        size_11mb = 11 * 1024 * 1024
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate("document.pdf", size_11mb)
        assert exc_info.value.code == "FILE_TOO_LARGE"

    def test_max_size_boundary(self):
        # Exactly at the limit should pass
        max_size = 10 * 1024 * 1024
        self.validator.validate("document.pdf", max_size)

    def test_pdf_uppercase_extension(self):
        # .PDF should also be accepted
        self.validator.validate("DOCUMENT.PDF", 1024)
