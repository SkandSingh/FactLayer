"""Round-robin pooling across multiple `LLMClient` instances.

Lets callers spread load across several API keys and/or providers so the
effective throughput isn't capped by any single key's rate limit.
"""
import asyncio
import logging

from app.llm.base import LLMClient

logger = logging.getLogger(__name__)


class RoundRobinLLMClient(LLMClient):
    """LLMClient that dispatches each call to the next client in a cycle.

    Wraps a fixed list of underlying `LLMClient` instances (e.g. one per
    API key, potentially across multiple providers) and round-robins
    calls across them, so concurrent load is spread roughly evenly and no
    single underlying client's rate limit is hit as quickly.

    Failover: if the chosen client's `generate` raises (its own internal
    retry loop, if any, is already exhausted by the time it does), this
    tries the NEXT client in the pool for the same call rather than
    failing the whole request outright -- a quota-exhausted or otherwise
    unhealthy key shouldn't sink a call that a different key/provider in
    the pool could have handled. Only raises once every client in the
    pool has failed for this call.
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

    async def _next_client(self) -> LLMClient:
        async with self._lock:
            client = self._clients[self._next_index]
            self._next_index = (self._next_index + 1) % len(self._clients)
        return client

    async def generate(self, prompt: str) -> str:
        last_exc: BaseException | None = None

        for _ in range(len(self._clients)):
            client = await self._next_client()
            try:
                return await client.generate(prompt)
            except Exception as exc:  # noqa: BLE001 - deliberately broad, see class docstring
                last_exc = exc
                logger.warning(
                    "Pooled LLM client %s failed, trying next client in "
                    "pool: %s",
                    type(client).__name__,
                    exc,
                )

        assert last_exc is not None  # at least one client always exists
        raise last_exc
