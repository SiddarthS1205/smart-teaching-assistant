"""
test_tools.py — Unit tests for all 7 MCP Tools.

Tests run with zero real API calls:
  - EmbeddingGenerator: OpenAI patched; sentence-transformers stubbed
  - LLMGenerator / Summarizer: OpenAI patched; free fallback exercised directly
  - VectorDatabase: real FAISS with tiny (dim=8) vectors
  - DocumentLoader: real PyPDF2 against an in-memory minimal PDF

Run:
    pytest tests/test_tools.py -v
"""
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# conftest puts backend/ on sys.path and sets env vars
from tests.conftest import make_pdf, EMBED_DIM


# ══════════════════════════════════════════════════════════════════════════════
# Tool 1 — DocumentLoader
# ══════════════════════════════════════════════════════════════════════════════

class TestDocumentLoader:
    """Tests for PDF loading and metadata extraction."""

    @pytest.fixture(autouse=True)
    def loader(self):
        from tools.document_loader import DocumentLoader
        self.loader = DocumentLoader()

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_load_returns_string(self, sample_pdf_path):
        """Loading a valid PDF returns a non-empty string."""
        text = self.loader.load(str(sample_pdf_path))
        assert isinstance(text, str)
        assert len(text) > 0

    def test_load_contains_expected_text(self, sample_pdf_path):
        """Extracted text contains words from the PDF content."""
        text = self.loader.load(str(sample_pdf_path))
        # At least some words from the embedded sentence should appear
        assert any(word in text for word in ["intelligence", "learning", "machine", "artificial"])

    def test_load_metadata_returns_dict(self, sample_pdf_path):
        """load_metadata returns a dict with required keys."""
        meta = self.loader.load_metadata(str(sample_pdf_path))
        assert isinstance(meta, dict)
        for key in ("filename", "pages", "size_bytes"):
            assert key in meta, f"Missing key: {key}"

    def test_load_metadata_page_count(self, sample_pdf_path):
        """Minimal PDF has exactly 1 page."""
        meta = self.loader.load_metadata(str(sample_pdf_path))
        assert meta["pages"] == 1

    def test_load_metadata_filename(self, sample_pdf_path):
        """Metadata filename matches the actual file name."""
        meta = self.loader.load_metadata(str(sample_pdf_path))
        assert meta["filename"] == "sample.pdf"

    def test_load_metadata_size_positive(self, sample_pdf_path):
        """File size reported in metadata is positive."""
        meta = self.loader.load_metadata(str(sample_pdf_path))
        assert meta["size_bytes"] > 0

    # ── Error handling ────────────────────────────────────────────────────────

    def test_load_missing_file_raises_file_not_found(self, tmp_path):
        """Loading a non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            self.loader.load(str(tmp_path / "ghost.pdf"))

    def test_load_non_pdf_raises_value_error(self, tmp_path):
        """Loading a .txt file raises ValueError."""
        txt = tmp_path / "doc.txt"
        txt.write_text("hello world")
        with pytest.raises(ValueError, match="PDF"):
            self.loader.load(str(txt))

    def test_load_wrong_extension_raises(self, tmp_path):
        """Any non-.pdf extension is rejected."""
        for ext in [".docx", ".png", ".csv", ".exe"]:
            f = tmp_path / f"file{ext}"
            f.write_bytes(b"dummy")
            with pytest.raises(ValueError):
                self.loader.load(str(f))


# ══════════════════════════════════════════════════════════════════════════════
# Tool 2 — TextChunker
# ══════════════════════════════════════════════════════════════════════════════

class TestTextChunker:
    """Tests for word-boundary text splitting."""

    @pytest.fixture(autouse=True)
    def chunker(self):
        from tools.text_chunker import TextChunker
        # Small sizes so tests are fast and predictable
        self.chunker = TextChunker(chunk_size=10, chunk_overlap=2)

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_chunk_returns_list(self):
        """chunk() always returns a list."""
        result = self.chunker.chunk("word " * 30)
        assert isinstance(result, list)

    def test_chunk_produces_multiple_chunks(self):
        """Long text produces more than one chunk."""
        text = " ".join(f"word{i}" for i in range(50))
        chunks = self.chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunk_size_not_exceeded(self):
        """No chunk contains more words than chunk_size."""
        text = " ".join(f"w{i}" for i in range(100))
        for chunk in self.chunker.chunk(text):
            assert len(chunk.split()) <= self.chunker.chunk_size

    def test_chunk_overlap_shared_words(self):
        """Consecutive chunks share words equal to the overlap setting."""
        chunker = __import__("tools.text_chunker", fromlist=["TextChunker"]).TextChunker(
            chunk_size=6, chunk_overlap=2
        )
        words = [f"w{i}" for i in range(20)]
        chunks = chunker.chunk(" ".join(words))
        if len(chunks) >= 2:
            tail = set(chunks[0].split()[-2:])
            head = set(chunks[1].split()[:2])
            assert tail & head, "Expected overlapping words between consecutive chunks"

    def test_chunk_all_words_preserved(self):
        """Every word in the original text appears in at least one chunk."""
        words = [f"unique{i}" for i in range(30)]
        text = " ".join(words)
        chunks = self.chunker.chunk(text)
        combined = " ".join(chunks)
        for w in words:
            assert w in combined

    def test_chunk_short_text_single_chunk(self):
        """Text shorter than chunk_size produces exactly one chunk."""
        text = "hello world this is short"
        chunks = self.chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_chunk_empty_string_returns_empty_list(self):
        """Empty string returns []."""
        assert self.chunker.chunk("") == []

    def test_chunk_whitespace_only_returns_empty_list(self):
        """Whitespace-only string returns []."""
        assert self.chunker.chunk("   \n\t  ") == []

    def test_chunk_single_word(self):
        """Single word produces one chunk."""
        chunks = self.chunker.chunk("hello")
        assert len(chunks) == 1
        assert chunks[0] == "hello"

    def test_chunk_exact_size_boundary(self):
        """Text with exactly chunk_size words produces one or two chunks (overlap creates a tail)."""
        text = " ".join(f"w{i}" for i in range(10))
        chunks = self.chunker.chunk(text)
        # With overlap=2, a 10-word text with chunk_size=10 may produce a small tail chunk
        assert len(chunks) >= 1
        # The first chunk must contain all 10 words
        assert len(chunks[0].split()) == 10


# ══════════════════════════════════════════════════════════════════════════════
# Tool 3 — EmbeddingGenerator
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingGenerator:
    """Tests for embedding generation (OpenAI mocked, local model stubbed)."""

    def _make_generator_with_mock_local(self, dim=384):
        """Return an EmbeddingGenerator whose local model is a MagicMock."""
        from tools.embedding_generator import EmbeddingGenerator
        gen = EmbeddingGenerator.__new__(EmbeddingGenerator)
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda texts, **kw: np.random.default_rng(0).random(
            (len(texts), dim)
        ).astype(np.float32)
        gen._use_local = True
        gen._local_model = mock_model
        gen._openai_client = None
        return gen

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_embed_returns_list_of_lists(self):
        """embed() returns a list of float lists."""
        gen = self._make_generator_with_mock_local()
        result = gen.embed(["hello world", "test sentence"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)

    def test_embed_correct_count(self):
        """Number of returned vectors equals number of input texts."""
        gen = self._make_generator_with_mock_local()
        texts = ["a", "b", "c", "d", "e"]
        result = gen.embed(texts)
        assert len(result) == len(texts)

    def test_embed_vector_dimension(self):
        """Each vector has the expected dimension."""
        dim = 384
        gen = self._make_generator_with_mock_local(dim=dim)
        result = gen.embed(["test"])
        assert len(result[0]) == dim

    def test_embed_single_returns_flat_list(self):
        """embed_single() returns a single flat list of floats."""
        gen = self._make_generator_with_mock_local()
        result = gen.embed_single("single text")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_embed_single_consistent_with_embed(self):
        """embed_single(t) == embed([t])[0] for the same input."""
        gen = self._make_generator_with_mock_local()
        # Use a fixed seed so both calls return the same vector
        gen._local_model.encode.side_effect = None
        gen._local_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        v1 = gen.embed(["hello"])[0]
        gen._local_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        v2 = gen.embed_single("hello")
        assert v1 == v2

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_embed_empty_list_returns_empty(self):
        """embed([]) returns []."""
        gen = self._make_generator_with_mock_local()
        assert gen.embed([]) == []

    def test_embed_no_backend_raises_runtime_error(self):
        """RuntimeError raised when no backend is available."""
        from tools.embedding_generator import EmbeddingGenerator
        gen = EmbeddingGenerator.__new__(EmbeddingGenerator)
        gen._use_local = False
        gen._local_model = None
        gen._openai_client = None
        with pytest.raises(RuntimeError, match="No embedding backend"):
            gen.embed(["test"])

    def test_embed_openai_path(self):
        """OpenAI path returns correct vectors when local is disabled."""
        from tools.embedding_generator import EmbeddingGenerator
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.5] * 1536)]

        mock_openai_client = MagicMock()
        mock_openai_client.embeddings.create.return_value = mock_response

        gen = EmbeddingGenerator.__new__(EmbeddingGenerator)
        gen._use_local = False
        gen._local_model = None
        gen._openai_client = mock_openai_client
        result = gen.embed(["test"])

        assert len(result) == 1
        assert len(result[0]) == 1536


# ══════════════════════════════════════════════════════════════════════════════
# Tool 4 — VectorDatabase
# ══════════════════════════════════════════════════════════════════════════════

class TestVectorDatabase:
    """Tests for FAISS vector storage, search, and persistence."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        from tools.vector_database import VectorDatabase
        self.db = VectorDatabase(index_dir=str(tmp_path / "faiss"))
        self.dim = EMBED_DIM
        rng = np.random.default_rng(7)
        self.embeddings = [rng.random(self.dim).tolist() for _ in range(6)]
        self.chunks = [f"chunk_{i}" for i in range(6)]
        self.tmp_path = tmp_path

    # ── State checks ──────────────────────────────────────────────────────────

    def test_not_ready_before_build(self):
        """is_ready() is False before build() is called."""
        assert not self.db.is_ready()

    def test_ready_after_build(self):
        """is_ready() is True after build()."""
        self.db.build(self.embeddings, self.chunks)
        assert self.db.is_ready()

    # ── Build ─────────────────────────────────────────────────────────────────

    def test_build_stores_correct_chunk_count(self):
        """After build, internal chunk list matches input."""
        self.db.build(self.embeddings, self.chunks)
        assert len(self.db._chunks) == len(self.chunks)

    def test_build_mismatched_lengths_raises(self):
        """build() raises ValueError when embeddings and chunks differ in length."""
        with pytest.raises(ValueError, match="same length"):
            self.db.build(self.embeddings[:3], self.chunks[:5])

    # ── Search ────────────────────────────────────────────────────────────────

    def test_search_returns_list(self):
        """search() returns a list."""
        self.db.build(self.embeddings, self.chunks)
        q = np.random.default_rng(0).random(self.dim).tolist()
        result = self.db.search(q, top_k=2)
        assert isinstance(result, list)

    def test_search_respects_top_k(self):
        """search() returns at most top_k results."""
        self.db.build(self.embeddings, self.chunks)
        q = np.random.default_rng(0).random(self.dim).tolist()
        for k in [1, 2, 3]:
            result = self.db.search(q, top_k=k)
            assert len(result) <= k

    def test_search_returns_strings(self):
        """All returned items are strings."""
        self.db.build(self.embeddings, self.chunks)
        q = np.random.default_rng(0).random(self.dim).tolist()
        for item in self.db.search(q, top_k=3):
            assert isinstance(item, str)

    def test_search_results_are_from_corpus(self):
        """Every returned chunk exists in the original corpus."""
        self.db.build(self.embeddings, self.chunks)
        q = np.random.default_rng(0).random(self.dim).tolist()
        for item in self.db.search(q, top_k=3):
            assert item in self.chunks

    def test_search_exact_vector_returns_itself(self):
        """Searching with an exact stored vector returns that chunk first."""
        self.db.build(self.embeddings, self.chunks)
        result = self.db.search(self.embeddings[0], top_k=1)
        assert result[0] == self.chunks[0]

    def test_search_before_build_raises(self):
        """search() raises RuntimeError when index is empty."""
        with pytest.raises(RuntimeError, match="empty"):
            self.db.search([0.1] * self.dim)

    # ── Persistence ───────────────────────────────────────────────────────────

    def test_persist_and_reload(self, tmp_path):
        """Index saved by build() can be reloaded into a fresh instance."""
        from tools.vector_database import VectorDatabase
        self.db.build(self.embeddings, self.chunks)

        db2 = VectorDatabase(index_dir=str(tmp_path / "faiss"))
        assert db2.load() is True
        assert db2.is_ready()
        assert db2._chunks == self.chunks

    def test_load_returns_false_when_no_files(self, tmp_path):
        """load() returns False when no persisted files exist."""
        from tools.vector_database import VectorDatabase
        fresh = VectorDatabase(index_dir=str(tmp_path / "empty_idx"))
        assert fresh.load() is False

    # ── Clear ─────────────────────────────────────────────────────────────────

    def test_clear_resets_ready_state(self):
        """clear() makes is_ready() return False."""
        self.db.build(self.embeddings, self.chunks)
        self.db.clear()
        assert not self.db.is_ready()

    def test_clear_empties_chunks(self):
        """clear() empties the internal chunk list."""
        self.db.build(self.embeddings, self.chunks)
        self.db.clear()
        assert self.db._chunks == []


# ══════════════════════════════════════════════════════════════════════════════
# Tool 5 — Retriever
# ══════════════════════════════════════════════════════════════════════════════

class TestRetriever:
    """Tests for semantic retrieval combining embedder + FAISS."""

    def test_retrieve_returns_list(self, retriever):
        """retrieve() returns a list."""
        result = retriever.retrieve("What is machine learning?")
        assert isinstance(result, list)

    def test_retrieve_returns_strings(self, retriever):
        """All retrieved items are strings."""
        for item in retriever.retrieve("deep learning"):
            assert isinstance(item, str)

    def test_retrieve_respects_top_k(self, retriever):
        """retrieve() returns at most top_k results."""
        result = retriever.retrieve("neural networks", top_k=2)
        assert len(result) <= 2

    def test_retrieve_calls_embedder(self, retriever, mock_embedder):
        """Retriever calls embed_single on the query."""
        retriever.retrieve("test query")
        mock_embedder.embed_single.assert_called_once_with("test query")

    def test_retrieve_empty_when_db_not_ready(self, mock_embedder, tmp_path):
        """Returns [] when vector DB has no index."""
        from tools.vector_database import VectorDatabase
        from tools.retriever import Retriever
        empty_db = VectorDatabase(index_dir=str(tmp_path / "empty"))
        r = Retriever(mock_embedder, empty_db)
        result = r.retrieve("anything")
        assert result == []

    def test_retrieve_results_from_corpus(self, retriever, sample_chunks):
        """Every retrieved chunk is from the original corpus."""
        results = retriever.retrieve("artificial intelligence")
        for item in results:
            assert item in sample_chunks


# ══════════════════════════════════════════════════════════════════════════════
# Tool 6 — LLMGenerator (free extraction path)
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMGenerator:
    """Tests for the LLMGenerator — exercises the free extraction fallback."""

    @pytest.fixture(autouse=True)
    def generator(self):
        from tools.llm_generator import LLMGenerator
        self.gen = LLMGenerator.__new__(LLMGenerator)
        self.gen._openai = None   # force free path

    # ── Output format ─────────────────────────────────────────────────────────

    def test_generate_returns_string(self):
        """generate() always returns a string."""
        result = self.gen.generate("What is the student ID?",
                                   ["Student ID : 26013526"])
        assert isinstance(result, str)

    def test_generate_label_value_format(self):
        """Free path returns 'Label : Value' format."""
        result = self.gen.generate("What is the student ID?",
                                   ["Student ID : 26013526"])
        assert ":" in result

    def test_generate_student_id_extracted(self):
        """Student ID is correctly extracted from chunk."""
        result = self.gen.generate(
            "What is the student ID?",
            ["Student ID : 26013526 Student Name : ARUN KUMAR"]
        )
        assert "26013526" in result

    def test_generate_parent_name_strips_null(self):
        """'/ null' suffix is removed from extracted value."""
        result = self.gen.generate(
            "What is the parent name?",
            ["Parent Name : SIVAKOZUNDU / null Mobile : 9876543210"]
        )
        assert "SIVAKOZUNDU" in result
        assert "null" not in result.lower()

    def test_generate_course_extracted(self):
        """Course field is extracted correctly."""
        result = self.gen.generate(
            "What is the course?",
            ["Course : BTECH - Artificial Intelligence and Data Science"]
        )
        assert "BTECH" in result

    def test_generate_payment_mode(self):
        """Payment mode is extracted correctly."""
        result = self.gen.generate(
            "What is the payment mode?",
            ["Amount : Rs. 45000 Mode : APP Receipt No : RCP-001"]
        )
        assert "APP" in result

    def test_generate_invoice_status(self):
        """Invoice status APPROVED is extracted."""
        result = self.gen.generate(
            "What is the invoice status?",
            ["INVOICE STATUS: APPROVED RISK LEVEL: LOW"]
        )
        assert "APPROVED" in result

    def test_generate_total_due(self):
        """Total due amount is extracted."""
        result = self.gen.generate(
            "What is the total due?",
            ["Subtotal: 8840.92 TOTAL DUE: 8491.90"]
        )
        assert "8491.90" in result

    def test_generate_no_context_returns_not_found(self):
        """Empty context returns NOT FOUND."""
        result = self.gen.generate("What is the student ID?", [])
        assert "NOT FOUND" in result or "couldn't find" in result.lower()

    def test_generate_different_queries_different_answers(self):
        """Two different queries on the same chunks produce different answers."""
        chunks = [
            "Invoice #: INV-2026-0042 Due Date: April 01, 2026",
            "TOTAL DUE: 8491.90 Currency: USD",
        ]
        a1 = self.gen.generate("What is the due date?", chunks)
        a2 = self.gen.generate("What is the total due?", chunks)
        assert a1 != a2

    # ── OpenAI path ───────────────────────────────────────────────────────────

    def test_generate_openai_path_called(self):
        """When OpenAI client is set, it is called."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Student ID : 99999"))]
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = mock_response

        self.gen._openai = mock_openai
        result = self.gen.generate("What is the student ID?", ["Student ID : 99999"])
        assert result == "Student ID : 99999"
        mock_openai.chat.completions.create.assert_called_once()

    def test_generate_openai_quota_error_falls_back(self):
        """RateLimitError from OpenAI triggers free fallback."""
        from openai import RateLimitError
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.side_effect = RateLimitError(
            "quota exceeded", response=MagicMock(status_code=429), body={}
        )
        self.gen._openai = mock_openai
        # Should not raise — falls back to free extractor
        result = self.gen.generate(
            "What is the student ID?",
            ["Student ID : 26013526"]
        )
        assert isinstance(result, str)
        assert len(result) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Tool 7 — Summarizer
# ══════════════════════════════════════════════════════════════════════════════

class TestSummarizer:
    """Tests for the Summarizer tool — exercises the free extractive path."""

    @pytest.fixture(autouse=True)
    def summarizer(self):
        from tools.summarizer import Summarizer
        self.s = Summarizer.__new__(Summarizer)
        self.s._openai = None   # force free path

    def test_summarize_returns_string(self, sample_chunks):
        """summarize() returns a string."""
        result = self.s.summarize(sample_chunks)
        assert isinstance(result, str)

    def test_summarize_non_empty(self, sample_chunks):
        """Summary is not empty."""
        result = self.s.summarize(sample_chunks)
        assert len(result) > 50

    def test_summarize_empty_chunks_returns_message(self):
        """Empty chunk list returns a 'no content' message."""
        result = self.s.summarize([])
        assert "no document" in result.lower() or "available" in result.lower()

    def test_summarize_contains_key_terms_section(self, sample_chunks):
        """Free summary includes a Key Terms section."""
        result = self.s.summarize(sample_chunks)
        assert "Key Terms" in result or "key" in result.lower()

    def test_summarize_contains_document_length(self, sample_chunks):
        """Free summary reports document length."""
        result = self.s.summarize(sample_chunks)
        assert "words" in result.lower() or "chunk" in result.lower()

    def test_summarize_single_chunk(self):
        """Summarizer handles a single-chunk document."""
        result = self.s.summarize(["Artificial intelligence is transforming education."])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_openai_called_when_available(self, sample_chunks):
        """When OpenAI is set, it is called."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="AI Summary"))]
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = mock_response

        self.s._openai = mock_openai
        result = self.s.summarize(sample_chunks)
        assert result == "AI Summary"
        mock_openai.chat.completions.create.assert_called_once()

    def test_summarize_openai_quota_falls_back(self, sample_chunks):
        """RateLimitError from OpenAI triggers free fallback."""
        from openai import RateLimitError
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.side_effect = RateLimitError(
            "quota", response=MagicMock(status_code=429), body={}
        )
        self.s._openai = mock_openai
        result = self.s.summarize(sample_chunks)
        assert isinstance(result, str)
        assert len(result) > 0
