"""Gemini implementation of the provider-agnostic `LLMClient` interface."""
import asyncio

from google import genai

from app import config
from app.llm.base import LLMClient


class GeminiClient(LLMClient):
    """LLMClient backed by Google's Gemini API via the `google-genai` SDK."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key if api_key is not None else config.GEMINI_API_KEY
        self._model = model if model is not None else config.GEMINI_MODEL
        self._client = genai.Client(api_key=self._api_key)

    async def generate(self, prompt: str) -> str:
        """Send prompt to Gemini, return the raw text response.

        This is a plain single call guarded by a timeout
        (`config.LLM_TIMEOUT_SECONDS`) — no retry/backoff here, that is a
        separate hardening concern layered on top by callers. Any failure
        (timeout, API error, etc.) is raised to the caller as-is.
        """
        aio_models = getattr(self._client, "aio", None)
        if aio_models is not None and hasattr(aio_models, "models"):
            # Preferred path: the SDK's native async surface.
            coro = aio_models.models.generate_content(
                model=self._model,
                contents=prompt,
            )
        else:
            # Fallback for older/newer SDKs without an async surface:
            # run the sync call in a worker thread.
            coro = asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model,
                contents=prompt,
            )

        response = await asyncio.wait_for(coro, timeout=config.LLM_TIMEOUT_SECONDS)
        return response.text
