"""Round-robin pooling across multiple `LLMClient` instances.

Lets callers spread load across several API keys and/or providers so the
effective throughput isn't capped by any single key's rate limit.
"""
import asyncio

from app.llm.base import LLMClient


class RoundRobinLLMClient(LLMClient):
    """LLMClient that dispatches each call to the next client in a cycle.

    Wraps a fixed list of underlying `LLMClient` instances (e.g. one per
    API key, potentially across multiple providers) and round-robins
    calls across them, so concurrent load is spread roughly evenly and no
    single underlying client's rate limit is hit as quickly.
    """

    def __init__(self, clients: list[LLMClient]):
        if not clients:
            raise ValueError(
                "RoundRobinLLMClient needs at least one underlying LLMClient"
            )
        self._clients = list(clients)
        self._next_index = 0
        # Protects `_next_index` so concurrent `generate` calls each get a
        # distinct, correctly-advanced client rather than racing on the
        # read-modify-write of the index.
        self._lock = asyncio.Lock()

    async def generate(self, prompt: str) -> str:
        async with self._lock:
            client = self._clients[self._next_index]
            self._next_index = (self._next_index + 1) % len(self._clients)
        return await client.generate(prompt)
