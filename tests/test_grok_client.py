"""Tests for app.llm.grok_client.GrokClient.

No real network calls are made -- `httpx.AsyncClient.post` is
monkeypatched with a fake coroutine, either returning a canned successful
response or raising `httpx.HTTPStatusError` a controlled number of times
before succeeding, mirroring the pattern in tests/test_gemini_client.py.

No pytest-asyncio plugin is installed in this environment (same
constraint as tests/test_gemini_client.py), so async code under test is
driven with `asyncio.run(...)` rather than `@pytest.mark.asyncio`.

Run: python -m pytest tests/test_grok_client.py -v
"""
import asyncio

import httpx
import pytest

from app import config
from app.llm.grok_client import GrokClient


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Replace asyncio.sleep with a no-op so backoff delays don't slow
    down (or otherwise affect) these tests."""

    async def _fake_sleep(_seconds):
        return None

    monkeypatch.setattr("app.llm.grok_client.asyncio.sleep", _fake_sleep)


@pytest.fixture
def client(monkeypatch):
    # LLM_MAX_RETRIES defaults to 3 via app.config, but pin it explicitly
    # here so this test doesn't silently change behavior if the default
    # in .env / config.py ever changes.
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 3)
    return GrokClient(api_key="fake-key", model="fake-model")


class _FakeResponse:
    """Stand-in for `httpx.Response` for a successful chat-completion call."""

    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _make_error_response(status_code: int = 429) -> httpx.Response:
    request = httpx.Request("POST", "https://api.x.ai/v1/chat/completions")
    return httpx.Response(status_code=status_code, request=request)


class _FlakyPost:
    """Callable test double standing in for `httpx.AsyncClient.post`.

    Raises `httpx.HTTPStatusError` (simulating a rate-limit/5xx failure)
    for the first `fail_times` invocations, then returns a fake successful
    response carrying `result` as the completion content.
    """

    def __init__(self, fail_times: int, result: str = "final answer"):
        self.fail_times = fail_times
        self.result = result
        self.call_count = 0

    async def __call__(self, *args, **kwargs):
        self.call_count += 1
        if self.call_count <= self.fail_times:
            response = _make_error_response()
            raise httpx.HTTPStatusError(
                f"simulated failure #{self.call_count}",
                request=response.request,
                response=response,
            )
        return _FakeResponse(self.result)


def test_generate_returns_text_from_successful_response(client, monkeypatch):
    flaky = _FlakyPost(fail_times=0, result="hello from grok")
    monkeypatch.setattr(httpx.AsyncClient, "post", flaky)

    result = asyncio.run(client.generate("some prompt"))

    assert result == "hello from grok"
    assert flaky.call_count == 1


def test_generate_retries_then_succeeds_after_http_errors(client, monkeypatch):
    flaky = _FlakyPost(fail_times=2, result="success after retries")
    monkeypatch.setattr(httpx.AsyncClient, "post", flaky)

    result = asyncio.run(client.generate("some prompt"))

    assert result == "success after retries"
    # 2 failures + 1 success = 3 attempts total.
    assert flaky.call_count == 3


def test_generate_raises_after_exhausting_all_retries(client, monkeypatch):
    async def _always_fails(*args, **kwargs):
        response = _make_error_response()
        raise httpx.HTTPStatusError(
            "always fails", request=response.request, response=response
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", _always_fails)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.generate("some prompt"))


def test_generate_retries_on_timeout(client, monkeypatch):
    call_count = 0

    async def _timeout_then_succeed(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.TimeoutException("simulated timeout")
        return _FakeResponse("recovered")

    monkeypatch.setattr(httpx.AsyncClient, "post", _timeout_then_succeed)

    result = asyncio.run(client.generate("some prompt"))

    assert result == "recovered"
    assert call_count == 2
