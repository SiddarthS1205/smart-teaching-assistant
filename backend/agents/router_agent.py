"""
Router Agent — decides which agent to invoke based on the user's query.

Routing logic:
  • Keywords like "summary", "summarize", "overview", "outline" → SummarizerAgent
  • Everything else → QAAgent
"""
import re

from .base_agent import BaseAgent
from .qa_agent import QAAgent
from .summarizer_agent import SummarizerAgent
from observability import get_logger

logger = get_logger(__name__)

# Keywords that trigger the Summarizer Agent
SUMMARY_KEYWORDS = re.compile(
    r"\b(summar(?:y|ize|ise)|overview|outline|brief|abstract|synopsis|recap|tldr|tl;dr)\b",
    re.IGNORECASE,
)


class RouterAgent(BaseAgent):
    """
    Meta-agent that routes queries to the appropriate specialist agent.

    Decision tree:
        query matches SUMMARY_KEYWORDS → SummarizerAgent
        otherwise                      → QAAgent
    """

    def __init__(self, qa_agent: QAAgent, summarizer_agent: SummarizerAgent):
        super().__init__("RouterAgent")
        self._qa = qa_agent
        self._summarizer = summarizer_agent

    def route(self, query: str) -> str:
        """Determine which agent should handle *query*."""
        if SUMMARY_KEYWORDS.search(query):
            self.logger.info("🔀 Routing to SummarizerAgent (keyword match)")
            return "summarizer"
        self.logger.info("🔀 Routing to QAAgent (default)")
        return "qa"

    def run(self, query: str, **kwargs) -> str:
        """
        Route *query* to the correct agent and return its response.

        Args:
            query: The user's input.

        Returns:
            Response from the selected agent.
        """
        agent_name = self.route(query)

        if agent_name == "summarizer":
            return self._summarizer.run(query, **kwargs)
        return self._qa.run(query, **kwargs)
