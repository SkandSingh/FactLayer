"""Provider-agnostic LLM client abstraction.

Concrete implementations (e.g. `GeminiClient`) implement `generate` to
send a prompt to a specific LLM provider and return the raw text
response, so callers don't need to know which provider is behind it.
"""
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base class for LLM provider clients."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Send prompt, return the raw text response."""
