"""
MCP Tool 5 — Retriever
Performs semantic search: embeds the query then queries the vector store.
"""
from .embedding_generator import EmbeddingGenerator
from .vector_database import VectorDatabase
from config import settings
from observability import get_logger, timed

logger = get_logger(__name__)


class Retriever:
    """Semantic retriever that combines embedding + FAISS search."""

    def __init__(self, embedder: EmbeddingGenerator, vector_db: VectorDatabase):
        self._embedder = embedder
        self._vector_db = vector_db
        logger.info("Retriever initialised")

    @timed(logger)
    def retrieve(self, query: str, top_k: int = settings.TOP_K_RESULTS) -> list[str]:
        """
        Retrieve the most relevant chunks for *query*.

        Args:
            query: User's natural-language question.
            top_k: Number of chunks to return.

        Returns:
            List of relevant text chunks.
        """
        logger.info("🔎 Retrieving context for query: %s…", query[:80])

        if not self._vector_db.is_ready():
            logger.warning("Vector DB not ready — no document uploaded yet")
            return []

        query_vector = self._embedder.embed_single(query)
        chunks = self._vector_db.search(query_vector, top_k=top_k)
        return chunks
