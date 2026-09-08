"""Convenience builder for the application's default pooled LLM client."""
from app.llm.base import LLMClient


def build_default_llm_client() -> LLMClient:
    """Build a `RoundRobinLLMClient` from every configured Gemini + Grok key.

    Reads `config.GEMINI_API_KEYS` and `config.GROK_API_KEYS` (each already
    falls back sensibly to legacy single-key vars / empty, see
    `app.config`), builds one client per key, and pools them behind a
    single round-robin `LLMClient` so callers don't need to know how many
    keys or providers are configured.

    Raises `ValueError` if no keys are configured at all -- there would be
    nothing to build a usable client from.
    """
    from app import config
    from app.llm.gemini_client import GeminiClient
    from app.llm.grok_client import GrokClient
    from app.llm.pool import RoundRobinLLMClient

    clients: list[LLMClient] = [
        GeminiClient(api_key=k) for k in config.GEMINI_API_KEYS
    ] + [GrokClient(api_key=k) for k in config.GROK_API_KEYS]

    if not clients:
        raise ValueError(
            "No LLM API keys configured (set GEMINI_API_KEYS/GEMINI_API_KEY "
            "and/or GROK_API_KEYS in .env)"
        )

    return RoundRobinLLMClient(clients)
