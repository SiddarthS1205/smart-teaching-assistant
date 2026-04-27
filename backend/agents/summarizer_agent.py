"""
Summarizer Agent — produces a structured summary of the uploaded document.
"""
import time

from .base_agent import BaseAgent
from tools.summarizer import Summarizer
from tools.vector_database import VectorDatabase
from observability import track_query


class SummarizerAgent(BaseAgent):
    """Agent that summarizes the entire uploaded document."""

    def __init__(self, summarizer: Summarizer, vector_db: VectorDatabase):
        super().__init__("SummarizerAgent")
        self._summarizer = summarizer
        self._vector_db = vector_db

    def run(self, query: str, **kwargs) -> str:
        """
        Summarize the document currently loaded in the vector store.

        Args:
            query: The user's request (used for logging/tracking).

        Returns:
            Structured document summary.
        """
        self.logger.info("📋 SummarizerAgent processing request…")
        start = time.perf_counter()

        if not self._vector_db.is_ready():
            return (
                "No document has been uploaded yet. "
                "Please upload a PDF document first, then request a summary."
            )

        # Access stored chunks directly from the vector DB
        chunks = self._vector_db._chunks
        summary = self._summarizer.summarize(chunks)

        duration_ms = (time.perf_counter() - start) * 1000
        track_query(query, summary, self.name, duration_ms)
        self.logger.info("✅ SummarizerAgent done in %.2f ms", duration_ms)
        return summary
