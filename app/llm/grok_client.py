"""xAI Grok implementation of the provider-agnostic `LLMClient` interface.

Grok's API is OpenAI-Chat-Completions-compatible, so this client talks to
it directly over HTTP via `httpx` rather than depending on a
provider-specific SDK.
"""
import asyncio
import logging

import httpx

from app import config
from app.llm.base import LLMClient

logger = logging.getLogger(__name__)

# Upper bound on the exponential backoff delay between retries, in seconds.
# Mirrors `app.llm.gemini_client._MAX_BACKOFF_SECONDS` so both providers
# back off identically under sustained failures.
_MAX_BACKOFF_SECONDS = 30

_GROK_CHAT_COMPLETIONS_URL = "https://api.x.ai/v1/chat/completions"


class GrokClient(LLMClient):
    """LLMClient backed by xAI's Grok API (OpenAI-compatible chat endpoint)."""

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or config.GROK_MODEL

    async def _call_once(self, prompt: str) -> str:
        """Make a single timed call to the underlying Grok API.

        Guarded by a per-call timeout (`config.LLM_TIMEOUT_SECONDS`); does
        not retry. Any failure (timeout, HTTP error, etc.) is raised as-is
        to the caller, which is `generate`'s retry loop.
        """
        async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(
                _GROK_CHAT_COMPLETIONS_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def generate(self, prompt: str) -> str:
        """Send prompt to Grok, return the raw text response.

        Each individual call is guarded by a timeout
        (`config.LLM_TIMEOUT_SECONDS`, see `_call_once`). On top of that,
        this method retries with exponential backoff on any failure --
        timeouts, transport errors, and especially rate-limit/429-shaped
        errors from the API (`httpx.HTTPStatusError` for 4xx/5xx responses
        via `raise_for_status()`, `httpx.TimeoutException` for timeouts)
        -- up to `config.LLM_MAX_RETRIES` times, sleeping `2 ** attempt`
        seconds (capped at `_MAX_BACKOFF_SECONDS`) between attempts. We
        deliberately catch broadly here, matching `GeminiClient.generate`,
        since any other transient failure -- a raw network error, a
        connection reset, etc. -- deserves the same retry treatment. If
        every attempt fails, the last exception is re-raised to the
        caller.
        """
        last_exc: BaseException | None = None
        max_attempts = config.LLM_MAX_RETRIES + 1

        for attempt in range(max_attempts):
            try:
                return await self._call_once(prompt)
            except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
                last_exc = exc
                is_last_attempt = attempt == max_attempts - 1
                if is_last_attempt:
                    logger.warning(
                        "Grok call failed on attempt %d/%d (final attempt, "
                        "giving up): %s",
                        attempt + 1,
                        max_attempts,
                        exc,
                    )
                    break

                delay = min(2**attempt, _MAX_BACKOFF_SECONDS)
                logger.warning(
                    "Grok call failed on attempt %d/%d: %s; retrying in %ds",
                    attempt + 1,
                    max_attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        assert last_exc is not None  # at least one attempt always runs
        raise last_exc
