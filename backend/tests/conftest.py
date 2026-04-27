"""
conftest.py — Shared pytest fixtures and configuration.

All tests import from here automatically. No real API calls are made —
OpenAI is fully mocked, sentence-transformers is patched with a fast
deterministic stub so tests run in milliseconds.
"""
import os
import sys
import shutil
import tempfile
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Make backend/ importable from any working directory ──────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Env vars must be set BEFORE any backend module is imported ────────────────
os.environ["OPENAI_API_KEY"]  = "test-key-not-real"
os.environ["APP_SECRET_KEY"]  = "test-secret"
os.environ["DEMO_USERNAME"]   = "admin"
os.environ["DEMO_PASSWORD"]   = "password123"
os.environ["CHUNK_SIZE"]      = "20"
os.environ["CHUNK_OVERLAP"]   = "3"
os.environ["TOP_K_RESULTS"]   = "3"
os.environ["LOG_LEVEL"]       = "WARNING"   # keep test output clean


# ── Embedding dimension used throughout tests ─────────────────────────────────
EMBED_DIM = 8   # tiny vectors — fast, no real model needed


# ── Minimal valid PDF builder ─────────────────────────────────────────────────
def make_pdf(text: str = "This is a test document about artificial intelligence.") -> bytes:
    """Return bytes of a minimal but valid single-page PDF containing *text*."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
         b"/MediaBox [0 0 612 792] /Contents 4 0 R "
         b"/Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"),
        (b"4 0 obj\n<< /Length " + str(len(stream)).encode()
         + b" >>\nstream\n" + stream + b"\nendstream\nendobj\n"),
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    header = b"%PDF-1.4\n"
    body   = b"".join(objects)
    offsets, pos = [], len(header)
    for obj in objects:
        offsets.append(pos); pos += len(obj)
    xref_offset = len(header) + len(body)
    xref = (b"xref\n" + f"0 {len(objects)+1}\n".encode()
            + b"0000000000 65535 f \n"
            + b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets))
    trailer = (f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\n"
               f"startxref\n{xref_offset}\n%%EOF\n").encode()
    return header + body + xref + trailer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tmp_dir_session():
    """A temporary directory that persists for the whole test session."""
    d = tempfile.mkdtemp(prefix="sta_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_dir(tmp_path):
    """Per-test temporary directory (pytest built-in tmp_path)."""
    return tmp_path


@pytest.fixture
def sample_pdf_path(tmp_path) -> Path:
    """Write a minimal PDF to disk and return its path."""
    p = tmp_path / "sample.pdf"
    p.write_bytes(make_pdf("Machine learning is a subset of artificial intelligence."))
    return p


@pytest.fixture
def sample_chunks() -> list[str]:
    """Representative document chunks for testing."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing enables computers to understand text.",
        "Reinforcement learning trains agents through rewards and penalties.",
        "Computer vision allows machines to interpret visual information.",
    ]


@pytest.fixture
def sample_embeddings(sample_chunks) -> list[list[float]]:
    """Deterministic fake embeddings matching sample_chunks length."""
    rng = np.random.default_rng(42)
    return [rng.random(EMBED_DIM).tolist() for _ in sample_chunks]


@pytest.fixture
def vector_db(tmp_path, sample_embeddings, sample_chunks):
    """A pre-built VectorDatabase instance ready for search."""
    from tools.vector_database import VectorDatabase
    db = VectorDatabase(index_dir=str(tmp_path / "idx"))
    db.build(sample_embeddings, sample_chunks)
    return db


@pytest.fixture
def mock_embedder(sample_embeddings):
    """EmbeddingGenerator stub — returns deterministic vectors, no model load."""
    embedder = MagicMock()
    embedder.embed.return_value = sample_embeddings
    embedder.embed_single.return_value = sample_embeddings[0]
    return embedder


@pytest.fixture
def mock_generator():
    """LLMGenerator stub — returns a fixed answer string."""
    gen = MagicMock()
    gen.generate.return_value = "Student ID : 26013526"
    return gen


@pytest.fixture
def mock_summarizer_tool():
    """Summarizer tool stub."""
    s = MagicMock()
    s.summarize.return_value = "## Document Summary\n\nKey topic: AI and machine learning."
    return s


@pytest.fixture
def retriever(mock_embedder, vector_db):
    """Real Retriever wired to mock embedder + real FAISS vector_db."""
    from tools.retriever import Retriever
    return Retriever(mock_embedder, vector_db)


@pytest.fixture
def qa_agent(retriever, mock_generator):
    """QAAgent wired to real retriever + mock generator."""
    from agents.qa_agent import QAAgent
    return QAAgent(retriever, mock_generator)


@pytest.fixture
def summarizer_agent(mock_summarizer_tool, vector_db):
    """SummarizerAgent wired to mock summarizer + real vector_db."""
    from agents.summarizer_agent import SummarizerAgent
    return SummarizerAgent(mock_summarizer_tool, vector_db)


@pytest.fixture
def router_agent(qa_agent, summarizer_agent):
    """RouterAgent wired to qa_agent + summarizer_agent."""
    from agents.router_agent import RouterAgent
    return RouterAgent(qa_agent, summarizer_agent)


@pytest.fixture(scope="module")
def api_client():
    """
    FastAPI TestClient with all heavy dependencies mocked.
    sentence-transformers and OpenAI are patched so the app
    starts instantly without downloading any models.
    """
    # Patch sentence-transformers at the module level before app import
    mock_st = MagicMock()
    mock_st.encode.side_effect = lambda texts, **kw: np.random.default_rng(0).random(
        (len(texts), 384)
    ).astype(np.float32)

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_st):
        with patch("tools.embedding_generator._LOCAL_MODEL", mock_st):
            with patch("tools.embedding_generator._USE_LOCAL", True):
                with patch("tools.llm_generator._OPENAI_CLIENT", None):
                    with patch("tools.summarizer._OPENAI_CLIENT", None):
                        from fastapi.testclient import TestClient
                        # Import app fresh
                        import importlib
                        import main as main_module
                        importlib.reload(main_module)
                        client = TestClient(main_module.app)
                        yield client
