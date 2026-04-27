"""
MCP Tool 4 — Vector Database
Stores and persists embeddings using FAISS for fast similarity search.
"""
import os
import pickle
from pathlib import Path

import faiss
import numpy as np

from config import settings
from observability import get_logger, timed

logger = get_logger(__name__)

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.pkl"


class VectorDatabase:
    """FAISS-backed vector store with persistence."""

    def __init__(self, index_dir: str = settings.INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index: faiss.IndexFlatL2 | None = None
        self._chunks: list[str] = []
        self._dim: int = 0
        logger.info("VectorDatabase initialised — store: %s", self.index_dir)

    # ── Build / populate ──────────────────────────────────────────────────────

    @timed(logger)
    def build(self, embeddings: list[list[float]], chunks: list[str]) -> None:
        """
        Build a new FAISS index from embeddings and store corresponding chunks.

        Args:
            embeddings: List of embedding vectors.
            chunks: Matching list of text chunks.
        """
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")

        vectors = np.array(embeddings, dtype=np.float32)
        self._dim = vectors.shape[1]
        self._index = faiss.IndexFlatL2(self._dim)
        self._index.add(vectors)
        self._chunks = chunks

        logger.info(
            "✅ FAISS index built — %d vectors, dim=%d",
            self._index.ntotal,
            self._dim,
        )
        self._persist()

    # ── Search ────────────────────────────────────────────────────────────────

    @timed(logger)
    def search(self, query_vector: list[float], top_k: int = settings.TOP_K_RESULTS) -> list[str]:
        """
        Return the *top_k* most similar chunks for a query vector.

        Args:
            query_vector: Embedding of the query.
            top_k: Number of results to return.

        Returns:
            List of matching text chunks.
        """
        if self._index is None:
            raise RuntimeError("Index is empty — call build() or load() first.")

        q = np.array([query_vector], dtype=np.float32)
        distances, indices = self._index.search(q, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append(self._chunks[idx])
            logger.debug("  Retrieved chunk %d (L2=%.4f): %s…", idx, dist, self._chunks[idx][:60])

        logger.info("🔍 Retrieved %d chunks for query", len(results))
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self) -> None:
        faiss.write_index(self._index, str(self.index_dir / INDEX_FILE))
        with open(self.index_dir / CHUNKS_FILE, "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info("💾 Index persisted to %s", self.index_dir)

    def load(self) -> bool:
        """Load a previously persisted index. Returns True on success."""
        idx_path = self.index_dir / INDEX_FILE
        chunks_path = self.index_dir / CHUNKS_FILE

        if not idx_path.exists() or not chunks_path.exists():
            logger.warning("No persisted index found at %s", self.index_dir)
            return False

        self._index = faiss.read_index(str(idx_path))
        with open(chunks_path, "rb") as f:
            self._chunks = pickle.load(f)

        logger.info(
            "✅ Index loaded — %d vectors from %s",
            self._index.ntotal,
            self.index_dir,
        )
        return True

    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    def clear(self) -> None:
        """Wipe the index and stored chunks."""
        self._index = None
        self._chunks = []
        for fname in [INDEX_FILE, CHUNKS_FILE]:
            p = self.index_dir / fname
            if p.exists():
                p.unlink()
        logger.info("🗑️  Vector database cleared")
