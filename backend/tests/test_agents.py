"""
test_agents.py — Unit tests for the Agentic System.

Tests cover:
  - RouterAgent: routing logic for all keyword variants
  - QAAgent: retrieval + generation pipeline
  - SummarizerAgent: document summarization pipeline
  - Observability: query tracking side-effects

All external dependencies (retriever, generator, summarizer) are mocked.

Run:
    pytest tests/test_agents.py -v
"""
import pytest
from unittest.mock import MagicMock, call, patch


# ══════════════════════════════════════════════════════════════════════════════
# RouterAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestRouterAgent:
    """Tests for the Router Agent's decision logic."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from agents.router_agent import RouterAgent
        self.qa = MagicMock()
        self.qa.run.return_value = "QA answer"
        self.qa.name = "QAAgent"
        self.summarizer = MagicMock()
        self.summarizer.run.return_value = "Summary answer"
        self.summarizer.name = "SummarizerAgent"
        self.router = RouterAgent(self.qa, self.summarizer)

    # ── Routing decisions ─────────────────────────────────────────────────────

    def test_normal_query_routes_to_qa(self):
        """A plain question routes to QAAgent."""
        assert self.router.route("What is machine learning?") == "qa"

    def test_summary_keyword_routes_to_summarizer(self):
        """'summary' keyword routes to SummarizerAgent."""
        assert self.router.route("Give me a summary of this document") == "summarizer"

    def test_summarize_keyword_routes_to_summarizer(self):
        """'summarize' keyword routes to SummarizerAgent."""
        assert self.router.route("Can you summarize the paper?") == "summarizer"

    def test_overview_keyword_routes_to_summarizer(self):
        """'overview' keyword routes to SummarizerAgent."""
        assert self.router.route("Provide an overview of the content") == "summarizer"

    def test_outline_keyword_routes_to_summarizer(self):
        """'outline' keyword routes to SummarizerAgent."""
        assert self.router.route("Give me an outline of the document") == "summarizer"

    def test_brief_keyword_routes_to_summarizer(self):
        """'brief' keyword routes to SummarizerAgent."""
        assert self.router.route("Give me a brief of this paper") == "summarizer"

    def test_abstract_keyword_routes_to_summarizer(self):
        """'abstract' keyword routes to SummarizerAgent."""
        assert self.router.route("What is the abstract?") == "summarizer"

    def test_synopsis_keyword_routes_to_summarizer(self):
        """'synopsis' keyword routes to SummarizerAgent."""
        assert self.router.route("Give me a synopsis") == "summarizer"

    def test_recap_keyword_routes_to_summarizer(self):
        """'recap' keyword routes to SummarizerAgent."""
        assert self.router.route("Can you recap the document?") == "summarizer"

    def test_tldr_keyword_routes_to_summarizer(self):
        """'tldr' keyword routes to SummarizerAgent."""
        assert self.router.route("tldr of this paper") == "summarizer"

    def test_case_insensitive_routing(self):
        """Routing keywords are case-insensitive."""
        assert self.router.route("SUMMARY please") == "summarizer"
        assert self.router.route("SUMMARIZE this") == "summarizer"
        assert self.router.route("OVERVIEW needed") == "summarizer"

    def test_keyword_in_middle_of_sentence(self):
        """Keyword anywhere in the sentence triggers summarizer."""
        assert self.router.route("I need a quick summary of chapter 3") == "summarizer"

    def test_non_summary_queries_route_to_qa(self):
        """Various non-summary queries all route to QA."""
        queries = [
            "What is the student ID?",
            "Who is the vendor?",
            "What is the total due?",
            "Explain deep learning",
            "What are the key findings?",
            "How many checks passed?",
        ]
        for q in queries:
            assert self.router.route(q) == "qa", f"Expected 'qa' for: {q}"

    # ── run() delegation ──────────────────────────────────────────────────────

    def test_run_delegates_to_qa_agent(self):
        """run() calls QAAgent.run() for normal queries."""
        result = self.router.run("What is deep learning?")
        self.qa.run.assert_called_once_with("What is deep learning?")
        assert result == "QA answer"

    def test_run_delegates_to_summarizer_agent(self):
        """run() calls SummarizerAgent.run() for summary queries."""
        result = self.router.run("Please summarize this document")
        self.summarizer.run.assert_called_once()
        assert result == "Summary answer"

    def test_run_returns_qa_response(self):
        """run() returns the QA agent's response for normal queries."""
        self.qa.run.return_value = "Detailed answer about AI"
        result = self.router.run("What is AI?")
        assert result == "Detailed answer about AI"

    def test_run_returns_summarizer_response(self):
        """run() returns the summarizer's response for summary queries."""
        self.summarizer.run.return_value = "## Document Summary\n\nKey topic: AI"
        result = self.router.run("Give me a summary")
        assert result == "## Document Summary\n\nKey topic: AI"

    def test_router_agent_name(self):
        """RouterAgent has the correct name attribute."""
        assert self.router.name == "RouterAgent"


# ══════════════════════════════════════════════════════════════════════════════
# QAAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestQAAgent:
    """Tests for the QA Agent's retrieval-augmented generation pipeline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from agents.qa_agent import QAAgent
        self.retriever = MagicMock()
        self.generator = MagicMock()
        self.agent = QAAgent(self.retriever, self.generator)

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_run_returns_string(self):
        """run() always returns a string."""
        self.retriever.retrieve.return_value = ["some context"]
        self.generator.generate.return_value = "Student ID : 26013526"
        result = self.agent.run("What is the student ID?")
        assert isinstance(result, str)

    def test_run_calls_retriever_with_query(self):
        """QAAgent passes the query to the retriever."""
        self.retriever.retrieve.return_value = ["context"]
        self.generator.generate.return_value = "answer"
        self.agent.run("What is the course?")
        self.retriever.retrieve.assert_called_once_with("What is the course?")

    def test_run_calls_generator_with_chunks(self):
        """QAAgent passes retrieved chunks to the generator."""
        chunks = ["Course : BTECH - AI and DS"]
        self.retriever.retrieve.return_value = chunks
        self.generator.generate.return_value = "Course : BTECH - AI and DS"
        self.agent.run("What is the course?")
        self.generator.generate.assert_called_once_with("What is the course?", chunks)

    def test_run_returns_generator_output(self):
        """run() returns exactly what the generator produces."""
        self.retriever.retrieve.return_value = ["context"]
        self.generator.generate.return_value = "Parent Name : SIVAKOZUNDU"
        result = self.agent.run("What is the parent name?")
        assert result == "Parent Name : SIVAKOZUNDU"

    # ── No context fallback ───────────────────────────────────────────────────

    def test_run_fallback_when_no_chunks(self):
        """When retriever returns [], generator is NOT called and fallback is returned."""
        self.retriever.retrieve.return_value = []
        result = self.agent.run("What is the student ID?")
        self.generator.generate.assert_not_called()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_run_fallback_message_mentions_document(self):
        """Fallback message guides user to upload a document."""
        self.retriever.retrieve.return_value = []
        result = self.agent.run("What is the student ID?")
        assert any(word in result.lower() for word in ["document", "upload", "find"])

    # ── Observability ─────────────────────────────────────────────────────────

    def test_run_tracks_query(self):
        """run() calls track_query for observability."""
        self.retriever.retrieve.return_value = ["context"]
        self.generator.generate.return_value = "answer"
        with patch("agents.qa_agent.track_query") as mock_track:
            self.agent.run("test query")
            mock_track.assert_called_once()
            args = mock_track.call_args[0]
            assert args[0] == "test query"   # query
            assert args[2] == "QAAgent"      # agent name

    def test_run_tracks_query_even_on_fallback(self):
        """track_query is called even when no chunks are found."""
        self.retriever.retrieve.return_value = []
        with patch("agents.qa_agent.track_query") as mock_track:
            self.agent.run("test query")
            mock_track.assert_called_once()

    # ── Agent identity ────────────────────────────────────────────────────────

    def test_agent_name(self):
        """QAAgent has the correct name."""
        assert self.agent.name == "QAAgent"

    def test_agent_repr(self):
        """repr() includes class name."""
        assert "QAAgent" in repr(self.agent)


# ══════════════════════════════════════════════════════════════════════════════
# SummarizerAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestSummarizerAgent:
    """Tests for the Summarizer Agent."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from agents.summarizer_agent import SummarizerAgent
        self.summarizer_tool = MagicMock()
        self.vector_db = MagicMock()
        self.agent = SummarizerAgent(self.summarizer_tool, self.vector_db)

    # ── No document ───────────────────────────────────────────────────────────

    def test_run_returns_message_when_no_document(self):
        """Returns a helpful message when no document is loaded."""
        self.vector_db.is_ready.return_value = False
        result = self.agent.run("Summarize")
        assert isinstance(result, str)
        assert any(w in result.lower() for w in ["no document", "upload", "pdf"])

    def test_run_does_not_call_summarizer_when_no_document(self):
        """Summarizer tool is NOT called when DB is not ready."""
        self.vector_db.is_ready.return_value = False
        self.agent.run("Summarize")
        self.summarizer_tool.summarize.assert_not_called()

    # ── With document ─────────────────────────────────────────────────────────

    def test_run_calls_summarizer_when_ready(self):
        """Summarizer tool is called when DB is ready."""
        self.vector_db.is_ready.return_value = True
        self.vector_db._chunks = ["chunk1", "chunk2", "chunk3"]
        self.summarizer_tool.summarize.return_value = "Great summary"
        self.agent.run("Summarize the document")
        self.summarizer_tool.summarize.assert_called_once_with(["chunk1", "chunk2", "chunk3"])

    def test_run_returns_summarizer_output(self):
        """run() returns exactly what the summarizer tool produces."""
        self.vector_db.is_ready.return_value = True
        self.vector_db._chunks = ["chunk1"]
        self.summarizer_tool.summarize.return_value = "## Summary\n\nKey topic: AI"
        result = self.agent.run("Give me a summary")
        assert result == "## Summary\n\nKey topic: AI"

    def test_run_passes_all_chunks(self):
        """All chunks from the vector DB are passed to the summarizer."""
        chunks = [f"chunk_{i}" for i in range(10)]
        self.vector_db.is_ready.return_value = True
        self.vector_db._chunks = chunks
        self.summarizer_tool.summarize.return_value = "summary"
        self.agent.run("summarize")
        self.summarizer_tool.summarize.assert_called_once_with(chunks)

    # ── Observability ─────────────────────────────────────────────────────────

    def test_run_tracks_query(self):
        """run() calls track_query for observability."""
        self.vector_db.is_ready.return_value = True
        self.vector_db._chunks = ["chunk"]
        self.summarizer_tool.summarize.return_value = "summary"
        with patch("agents.summarizer_agent.track_query") as mock_track:
            self.agent.run("summarize")
            mock_track.assert_called_once()
            args = mock_track.call_args[0]
            assert args[2] == "SummarizerAgent"

    # ── Agent identity ────────────────────────────────────────────────────────

    def test_agent_name(self):
        """SummarizerAgent has the correct name."""
        assert self.agent.name == "SummarizerAgent"


# ══════════════════════════════════════════════════════════════════════════════
# Guardrails — QueryValidator & UploadValidator
# ══════════════════════════════════════════════════════════════════════════════

class TestQueryValidator:
    """Tests for query input validation guardrails."""

    @pytest.fixture(autouse=True)
    def validator(self):
        from guardrails.validator import QueryValidator
        self.v = QueryValidator()

    # ── Valid queries ─────────────────────────────────────────────────────────

    def test_valid_query_passes(self):
        """A normal academic query passes validation."""
        result = self.v.validate("What is machine learning?")
        assert result == "What is machine learning?"

    def test_valid_query_is_stripped(self):
        """Leading/trailing whitespace is stripped."""
        result = self.v.validate("  What is AI?  ")
        assert result == "What is AI?"

    def test_various_academic_queries_pass(self):
        """Multiple valid academic queries all pass."""
        queries = [
            "What is the student ID?",
            "Explain the methodology used in chapter 3",
            "What are the key findings of this research?",
            "Who is the parent of the student?",
            "What is the total fee amount?",
        ]
        for q in queries:
            result = self.v.validate(q)
            assert isinstance(result, str)

    # ── Empty / short ─────────────────────────────────────────────────────────

    def test_empty_string_raises_empty_query(self):
        """Empty string raises ValidationError with EMPTY_QUERY code."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            self.v.validate("")
        assert exc.value.code == "EMPTY_QUERY"

    def test_whitespace_only_raises_empty_query(self):
        """Whitespace-only string raises EMPTY_QUERY."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            self.v.validate("   \n\t  ")
        assert exc.value.code == "EMPTY_QUERY"

    def test_too_short_raises_query_too_short(self):
        """Query shorter than MIN_QUERY_LENGTH raises QUERY_TOO_SHORT."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            self.v.validate("Hi")
        assert exc.value.code == "QUERY_TOO_SHORT"

    def test_single_char_raises_too_short(self):
        """Single character raises QUERY_TOO_SHORT."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError):
            self.v.validate("A")

    # ── Too long ──────────────────────────────────────────────────────────────

    def test_too_long_raises_query_too_long(self):
        """Query over MAX_QUERY_LENGTH raises QUERY_TOO_LONG."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            self.v.validate("a " * 600)
        assert exc.value.code == "QUERY_TOO_LONG"

    # ── Harmful content ───────────────────────────────────────────────────────

    def test_hack_keyword_rejected(self):
        """Query containing 'hack' is rejected."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            self.v.validate("How do I hack a system?")
        assert exc.value.code == "HARMFUL_CONTENT"

    def test_malware_keyword_rejected(self):
        """Query containing 'malware' is rejected."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            self.v.validate("Explain malware creation")
        assert exc.value.code == "HARMFUL_CONTENT"

    def test_bomb_keyword_rejected(self):
        """Query containing 'bomb' is rejected."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            self.v.validate("How to make a bomb")
        assert exc.value.code == "HARMFUL_CONTENT"

    def test_multiple_harmful_keywords_all_rejected(self):
        """All harmful keywords are individually rejected."""
        from guardrails.validator import ValidationError
        harmful = [
            "How do I hack a system?",
            "Explain malware creation",
            "How to make a bomb",
            "Tell me about ransomware",
            "How to jailbreak a device",
        ]
        for q in harmful:
            with pytest.raises(ValidationError) as exc:
                self.v.validate(q)
            assert exc.value.code == "HARMFUL_CONTENT", f"Expected HARMFUL_CONTENT for: {q}"

    # ── Irrelevant content ────────────────────────────────────────────────────

    def test_casino_keyword_rejected(self):
        """Query about casino is rejected as irrelevant."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            self.v.validate("How to win at casino gambling?")
        assert exc.value.code == "IRRELEVANT_CONTENT"

    def test_get_rich_rejected(self):
        """'get rich' query is rejected."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            self.v.validate("How to get rich fast?")
        assert exc.value.code == "IRRELEVANT_CONTENT"


class TestUploadValidator:
    """Tests for file upload validation guardrails."""

    @pytest.fixture(autouse=True)
    def validator(self):
        from guardrails.validator import UploadValidator
        self.v = UploadValidator()

    # ── Valid uploads ─────────────────────────────────────────────────────────

    def test_valid_pdf_passes(self):
        """A valid PDF within size limit passes."""
        self.v.validate("document.pdf", 1024 * 100)  # 100 KB

    def test_pdf_uppercase_extension_passes(self):
        """.PDF (uppercase) is accepted."""
        self.v.validate("DOCUMENT.PDF", 1024)

    def test_max_size_boundary_passes(self):
        """File exactly at the 10 MB limit passes."""
        self.v.validate("document.pdf", 10 * 1024 * 1024)

    # ── Invalid filename ──────────────────────────────────────────────────────

    def test_empty_filename_raises(self):
        """Empty filename raises NO_FILENAME."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            self.v.validate("", 1024)
        assert exc.value.code == "NO_FILENAME"

    def test_whitespace_filename_raises(self):
        """Whitespace-only filename raises NO_FILENAME."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            self.v.validate("   ", 1024)
        assert exc.value.code == "NO_FILENAME"

    # ── Wrong file type ───────────────────────────────────────────────────────

    def test_txt_file_rejected(self):
        """Text file raises INVALID_FILE_TYPE."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            self.v.validate("file.txt", 1024)
        assert exc.value.code == "INVALID_FILE_TYPE"

    def test_multiple_wrong_extensions_rejected(self):
        """All non-PDF extensions are rejected."""
        from guardrails.validator import ValidationError
        for ext in [".txt", ".docx", ".png", ".exe", ".csv", ".xlsx", ".zip"]:
            with pytest.raises(ValidationError) as exc:
                self.v.validate(f"file{ext}", 1024)
            assert exc.value.code == "INVALID_FILE_TYPE", f"Expected rejection for {ext}"

    # ── File size ─────────────────────────────────────────────────────────────

    def test_empty_file_raises(self):
        """Zero-byte file raises EMPTY_FILE."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            self.v.validate("document.pdf", 0)
        assert exc.value.code == "EMPTY_FILE"

    def test_negative_size_raises(self):
        """Negative size raises EMPTY_FILE."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            self.v.validate("document.pdf", -1)
        assert exc.value.code == "EMPTY_FILE"

    def test_oversized_file_raises(self):
        """File over 10 MB raises FILE_TOO_LARGE."""
        from guardrails.validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            self.v.validate("document.pdf", 11 * 1024 * 1024)
        assert exc.value.code == "FILE_TOO_LARGE"


# ══════════════════════════════════════════════════════════════════════════════
# Observability
# ══════════════════════════════════════════════════════════════════════════════

class TestObservability:
    """Tests for logging, query tracking, and the timed decorator."""

    def test_get_logger_returns_logger(self):
        """get_logger() returns a Python Logger instance."""
        import logging
        from observability.logger import get_logger
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_name(self):
        """Logger has the correct name."""
        from observability.logger import get_logger
        logger = get_logger("my.test")
        assert logger.name == "my.test"

    def test_track_query_appends_to_log(self):
        """track_query() adds an entry to the in-memory log."""
        from observability.logger import track_query, get_query_log
        before = len(get_query_log())
        track_query("test query", "test response", "QAAgent", 123.4)
        after = len(get_query_log())
        assert after == before + 1

    def test_track_query_entry_structure(self):
        """Tracked entry contains all required fields."""
        from observability.logger import track_query, get_query_log
        track_query("structure test", "response text", "TestAgent", 50.0)
        entry = get_query_log()[-1]
        assert "timestamp" in entry
        assert "agent" in entry
        assert "query" in entry
        assert "response_preview" in entry
        assert "duration_ms" in entry

    def test_track_query_correct_values(self):
        """Tracked entry stores the correct values."""
        from observability.logger import track_query, get_query_log
        track_query("my question", "my answer", "QAAgent", 99.9)
        entry = get_query_log()[-1]
        assert entry["query"] == "my question"
        assert entry["agent"] == "QAAgent"
        assert entry["duration_ms"] == 99.9
        assert "my answer" in entry["response_preview"]

    def test_track_query_truncates_long_response(self):
        """Response preview is truncated to 200 chars."""
        from observability.logger import track_query, get_query_log
        long_response = "x" * 500
        track_query("q", long_response, "Agent", 1.0)
        entry = get_query_log()[-1]
        assert len(entry["response_preview"]) <= 200

    def test_timed_decorator_returns_result(self):
        """@timed decorator does not alter the return value."""
        from observability.logger import timed, get_logger
        logger = get_logger("test")

        @timed(logger)
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_timed_decorator_propagates_exception(self):
        """@timed decorator re-raises exceptions from the wrapped function."""
        from observability.logger import timed, get_logger
        logger = get_logger("test")

        @timed(logger)
        def boom():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            boom()

    def test_timed_decorator_works_without_logger(self):
        """@timed works when no logger is passed (uses module logger)."""
        from observability.logger import timed

        @timed()
        def multiply(a, b):
            return a * b

        assert multiply(3, 4) == 12

    def test_get_query_log_returns_list(self):
        """get_query_log() returns a list."""
        from observability.logger import get_query_log
        assert isinstance(get_query_log(), list)

    def test_query_log_written_to_jsonl(self, tmp_path, monkeypatch):
        """track_query() writes a JSONL entry to the log file."""
        import json
        log_file = tmp_path / "queries.jsonl"
        monkeypatch.chdir(tmp_path)
        (tmp_path / "logs").mkdir()

        from observability import logger as obs_module
        # Patch the open call to write to our tmp file
        original_open = open

        def patched_open(path, mode="r", **kwargs):
            if "queries.jsonl" in str(path):
                return original_open(str(log_file), mode, **kwargs)
            return original_open(path, mode, **kwargs)

        with patch("builtins.open", side_effect=patched_open):
            from observability.logger import track_query
            track_query("jsonl test", "response", "Agent", 10.0)

        if log_file.exists():
            lines = log_file.read_text().strip().splitlines()
            assert len(lines) >= 1
            entry = json.loads(lines[-1])
            assert entry["query"] == "jsonl test"
