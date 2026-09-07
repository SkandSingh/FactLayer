"""Tests for app.llm.gemini_client.GeminiClient's retry/backoff behavior.

No real network calls are made -- `GeminiClient._call_once` (the single
timed call to the underlying `google-genai` SDK) is monkeypatched with a
fake coroutine that fails a controlled number of times before succeeding
(or always fails), and `asyncio.sleep` is monkeypatched to a no-op so the
tests don't actually wait through the exponential backoff delays.

No pytest-asyncio plugin is installed in this environment (same
constraint as tests/test_extraction.py), so async code under test is
driven with `asyncio.run(...)` rather than `@pytest.mark.asyncio`.

Run: python -m pytest tests/test_gemini_client.py -v
"""
import asyncio

import pytest

from app import config
from app.llm.gemini_client import GeminiClient


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Replace asyncio.sleep with a no-op so backoff delays don't slow
    down (or otherwise affect) these tests."""

    async def _fake_sleep(_seconds):
        return None

    monkeypatch.setattr("app.llm.gemini_client.asyncio.sleep", _fake_sleep)


@pytest.fixture
def client(monkeypatch):
    # LLM_MAX_RETRIES defaults to 3 via app.config, but pin it explicitly
    # here so this test doesn't silently change behavior if the default
    # in .env / config.py ever changes.
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 3)
    return GeminiClient(api_key="fake-key", model="fake-model")


class _FlakyCall:
    """Callable test double standing in for `GeminiClient._call_once`.

    Raises `RuntimeError` (simulating a rate-limit/transport failure) for
    the first `fail_times` invocations, then returns `result`.
    """

    def __init__(self, fail_times: int, result: str = "final answer"):
        self.fail_times = fail_times
        self.result = result
        self.call_count = 0

    async def __call__(self, prompt: str) -> str:
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise RuntimeError(f"simulated failure #{self.call_count}")
        return self.result


class _AlwaysFails:
    def __init__(self):
        self.call_count = 0

    async def __call__(self, prompt: str) -> str:
        self.call_count += 1
        raise RuntimeError(f"simulated failure #{self.call_count}")


def test_generate_retries_then_succeeds(client, monkeypatch):
    flaky = _FlakyCall(fail_times=2, result="success after retries")
    monkeypatch.setattr(client, "_call_once", flaky)

    result = asyncio.run(client.generate("some prompt"))

    assert result == "success after retries"
    # 2 failures + 1 success = 3 attempts total.
    assert flaky.call_count == 3


def test_generate_succeeds_on_first_attempt_without_retrying(client, monkeypatch):
    flaky = _FlakyCall(fail_times=0, result="first try")
    monkeypatch.setattr(client, "_call_once", flaky)

    result = asyncio.run(client.generate("some prompt"))

    assert result == "first try"
    assert flaky.call_count == 1


def test_generate_raises_after_exhausting_all_retries(client, monkeypatch):
    always_fails = _AlwaysFails()
    monkeypatch.setattr(client, "_call_once", always_fails)

    with pytest.raises(RuntimeError, match="simulated failure"):
        asyncio.run(client.generate("some prompt"))

    # LLM_MAX_RETRIES=3 retries on top of the initial attempt == 4 total.
    assert always_fails.call_count == config.LLM_MAX_RETRIES + 1


def test_generate_raises_the_last_exception(client, monkeypatch):
    call_count = 0

    async def _fail_with_distinct_messages(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        raise RuntimeError(f"failure #{call_count}")

    monkeypatch.setattr(client, "_call_once", _fail_with_distinct_messages)

    with pytest.raises(RuntimeError, match=f"failure #{config.LLM_MAX_RETRIES + 1}"):
        asyncio.run(client.generate("some prompt"))
