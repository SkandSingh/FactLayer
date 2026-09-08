"""Tests for app.llm.pool.RoundRobinLLMClient.

Uses tiny fake `LLMClient` doubles (no real network calls) to verify
round-robin distribution, empty-list validation, single-client behavior,
and thread-safety of the shared index under concurrent async calls.

No pytest-asyncio plugin is installed in this environment (same
constraint as tests/test_gemini_client.py), so async code under test is
driven with `asyncio.run(...)` rather than `@pytest.mark.asyncio`.

Run: python -m pytest tests/test_llm_pool.py -v
"""
import asyncio

import pytest

from app.llm.base import LLMClient
from app.llm.pool import RoundRobinLLMClient


class _FakeLLMClient(LLMClient):
    """Test double that returns a fixed string and records call count/order."""

    def __init__(self, name: str, response: str | None = None, delay: float = 0.0):
        self.name = name
        self.response = response if response is not None else f"response-from-{name}"
        self.call_count = 0
        self.delay = delay

    async def generate(self, prompt: str) -> str:
        if self.delay:
            await asyncio.sleep(self.delay)
        self.call_count += 1
        return self.response


def test_round_robin_distributes_evenly_across_three_clients():
    clients = [_FakeLLMClient("a"), _FakeLLMClient("b"), _FakeLLMClient("c")]
    pool = RoundRobinLLMClient(clients)

    results = asyncio.run(_generate_n_times_sequentially(pool, 6))

    # 6 calls over 3 clients, in round-robin order: a, b, c, a, b, c.
    assert results == [
        "response-from-a",
        "response-from-b",
        "response-from-c",
        "response-from-a",
        "response-from-b",
        "response-from-c",
    ]
    assert clients[0].call_count == 2
    assert clients[1].call_count == 2
    assert clients[2].call_count == 2


async def _generate_n_times_sequentially(pool: RoundRobinLLMClient, n: int) -> list[str]:
    return [await pool.generate(f"prompt {i}") for i in range(n)]


def test_empty_client_list_raises():
    with pytest.raises(ValueError, match="at least one"):
        RoundRobinLLMClient([])


def test_single_client_receives_all_calls():
    client = _FakeLLMClient("solo")
    pool = RoundRobinLLMClient([client])

    results = asyncio.run(_generate_n_times_sequentially(pool, 5))

    assert results == ["response-from-solo"] * 5
    assert client.call_count == 5


def test_concurrent_calls_distribute_correctly_without_races():
    # A small artificial delay increases the odds of exposing a race on
    # the shared `_next_index` if the lock weren't there: without the
    # lock, concurrent tasks could all read the same index before any of
    # them advances it.
    clients = [_FakeLLMClient("a", delay=0.01), _FakeLLMClient("b", delay=0.01)]
    pool = RoundRobinLLMClient(clients)

    async def _run_concurrently():
        return await asyncio.gather(*[pool.generate(f"prompt {i}") for i in range(10)])

    results = asyncio.run(_run_concurrently())

    assert len(results) == 10
    total_calls = clients[0].call_count + clients[1].call_count
    assert total_calls == 10
    # With a correctly-alternating round robin over an even number of
    # calls, each client gets exactly half.
    assert clients[0].call_count == 5
    assert clients[1].call_count == 5
