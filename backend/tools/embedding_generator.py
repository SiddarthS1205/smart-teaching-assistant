"""
MCP Tool 3 — Embedding Generator
Converts text chunks into dense vector embeddings.

Strategy (auto-detected at startup):
  1. sentence-transformers (free, local, no API key needed)  ← preferred
  2. OpenAI text-embedding-3-small (requires API key + credits)  ← fallback
"""
import os
import numpy as np

from config import settings
from observability import get_logger, timed

logger = get_logger(__name__)

# ── Detect which backend to use ───────────────────────────────────────────────
_USE_LOCAL = False
_LOCAL_MODEL = None
_OPENAI_CLIENT = None

try:
    from sentence_transformers import SentenceTransformer
    _LOCAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    _USE_LOCAL = True
    logger.info("EmbeddingGenerator: using local model 'all-MiniLM-L6-v2' (free, no API key)")
except Exception:
    try:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI(api_key=settings.OPENAI_API_KEY)
        _USE_LOCAL = False
        logger.info("EmbeddingGenerator: using OpenAI 'text-embedding-3-small'")
    except Exception as e:
        logger.error("EmbeddingGenerator: no embedding backend available — %s", e)


class EmbeddingGenerator:
    """Generate embeddings using the best available backend."""

    def __init__(self):
        self._use_local = _USE_LOCAL
        self._local_model = _LOCAL_MODEL
        self._openai_client = _OPENAI_CLIENT
        backend = "local (sentence-transformers)" if self._use_local else "OpenAI"
        logger.info("EmbeddingGenerator initialised — backend: %s", backend)

    @timed(logger)
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning("embed() called with empty list")
            return []

        logger.info("🔢 Generating embeddings for %d chunks…", len(texts))

        if self._use_local and self._local_model is not None:
            return self._embed_local(texts)
        elif self._openai_client is not None:
            return self._embed_openai(texts)
        else:
            raise RuntimeError(
                "No embedding backend available. "
                "Install sentence-transformers (pip install sentence-transformers) "
                "or add a valid OpenAI API key with credits."
            )

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Use sentence-transformers (free, runs locally)."""
        vectors = self._local_model.encode(texts, show_progress_bar=False)
        result = [v.tolist() for v in vectors]
        logger.info("✅ Local embeddings generated — dim=%d", len(result[0]) if result else 0)
        return result

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Use OpenAI embeddings API."""
        cleaned = [t.replace("\n", " ") for t in texts]
        response = self._openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=cleaned,
        )
        vectors = [item.embedding for item in response.data]
        logger.info("✅ OpenAI embeddings generated — dim=%d", len(vectors[0]) if vectors else 0)
        return vectors

    def embed_single(self, text: str) -> list[float]:
        """Convenience wrapper for a single string."""
        return self.embed([text])[0]
