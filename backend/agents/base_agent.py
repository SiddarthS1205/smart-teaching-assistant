"""
Base Agent — defines the interface all agents must implement.
"""
from abc import ABC, abstractmethod
from observability import get_logger


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"agent.{name}")
        self.logger.info("Agent '%s' initialised", name)

    @abstractmethod
    def run(self, query: str, **kwargs) -> str:
        """
        Execute the agent's task.

        Args:
            query: The user's input.
            **kwargs: Additional context (e.g., chunks).

        Returns:
            Agent's response string.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
