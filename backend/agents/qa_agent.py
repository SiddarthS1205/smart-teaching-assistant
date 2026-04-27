"""
QA Agent — answers questions using RAG (Retriever + LLM Generator).
"""
import time

from .base_agent import BaseAgent
from tools.retriever import Retriever
from tools.llm_generator import LLMGenerator
from observability import track_query


class QAAgent(BaseAgent):
    """Retrieval-Augmented Generation agent for question answering."""

    def __init__(self, retriever: Retriever, generator: LLMGenerator):
        super().__init__("QAAgent")
        self._retriever = retriever
        self._generator = generator

    def run(self, query: str, **kwargs) -> str:
        """
        Answer *query* by retrieving relevant context and generating a response.

        Args:
            query: The student's question.

        Returns:
            Generated answer grounded in document context.
        """
        self.logger.info("🤖 QAAgent processing: %s…", query[:80])
        start = time.perf_counter()

        # Step 1 — Retrieve relevant chunks
        chunks = self._retriever.retrieve(query)

        if not chunks:
            self.logger.warning("No relevant chunks found for query")
            answer = (
                "I couldn't find relevant information in the uploaded document "
                "to answer your question. Please make sure a document has been "
                "uploaded and try rephrasing your question."
            )
        else:
            # Step 2 — Generate answer from context
            answer = self._generator.generate(query, chunks)

        duration_ms = (time.perf_counter() - start) * 1000
        track_query(query, answer, self.name, duration_ms)
        self.logger.info("✅ QAAgent done in %.2f ms", duration_ms)
        return answer
