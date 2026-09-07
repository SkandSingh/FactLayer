"""Gemini implementation of the provider-agnostic `LLMClient` interface."""
import asyncio
import logging

from google import genai

from app import config
from app.llm.base import LLMClient

logger = logging.getLogger(__name__)

# Upper bound on the exponential backoff delay between retries, in seconds.
# Without a cap, `2 ** attempt` would grow unboundedly for a generous
# LLM_MAX_RETRIES configuration; 30s is a reasonable ceiling that still
# gives a rate-limited API meaningful breathing room between attempts.
_MAX_BACKOFF_SECONDS = 30


class GeminiClient(LLMClient):
    """LLMClient backed by Google's Gemini API via the `google-genai` SDK."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key if api_key is not None else config.GEMINI_API_KEY
        self._model = model if model is not None else config.GEMINI_MODEL
        self._client = genai.Client(api_key=self._api_key)

    async def _call_once(self, prompt: str) -> str:
        """Make a single timed call to the underlying Gemini API.

        Guarded by a per-call timeout (`config.LLM_TIMEOUT_SECONDS`); does
        not retry. Any failure (timeout, API error, etc.) is raised as-is
        to the caller, which is `generate`'s retry loop.
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

    async def generate(self, prompt: str) -> str:
        """Send prompt to Gemini, return the raw text response.

        Each individual call is guarded by a timeout
        (`config.LLM_TIMEOUT_SECONDS`, see `_call_once`). On top of that,
        this method retries with exponential backoff on any failure --
        timeouts, transport errors, and especially rate-limit/429-shaped
        errors from the API (the `google-genai` SDK raises
        `google.genai.errors.ClientError` for 4xx responses, including
        429, and `ServerError` for 5xx; both are subclasses of the SDK's
        `APIError`, but we deliberately catch broadly here since any
        other transient failure -- a raw network error, a timeout, etc.
        -- deserves the same retry treatment) -- up to
        `config.LLM_MAX_RETRIES` times, sleeping `2 ** attempt` seconds
        (capped at `_MAX_BACKOFF_SECONDS`) between attempts. If every
        attempt fails, the last exception is re-raised to the caller.
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
                        "Gemini call failed on attempt %d/%d (final attempt, "
                        "giving up): %s",
                        attempt + 1,
                        max_attempts,
                        exc,
                    )
                    break

                delay = min(2**attempt, _MAX_BACKOFF_SECONDS)
                logger.warning(
                    "Gemini call failed on attempt %d/%d: %s; retrying in %ds",
                    attempt + 1,
                    max_attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        assert last_exc is not None  # at least one attempt always runs
        raise last_exc
