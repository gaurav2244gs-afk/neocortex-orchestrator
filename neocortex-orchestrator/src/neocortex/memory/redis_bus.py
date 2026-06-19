"""
Redis event bus.

Every agent-graph node publishes a small JSON event here as it runs --
which node fired, how long it took, the resulting confidence/drift score,
any anomaly flags, token usage. The FastAPI WebSocket endpoint subscribes
to the global channel and re-broadcasts to every connected dashboard
client, which is what makes the dashboard "real-time" rather than
poll-based.

Redis also backs lightweight per-session state (`save_session_state`)
so a session's history survives across requests/workers -- this is the
piece that lets the API scale horizontally behind multiple uvicorn
workers instead of keeping agent state in a single process's memory.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator

import redis.asyncio as redis


@dataclass
class AgentEvent:
    session_id: str
    node: str
    event_type: str  # "node_start" | "node_end" | "anomaly" | "session_complete"
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class EventBus:
    def __init__(self, redis_url: str, channel: str = "neocortex:events"):
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self.channel = channel

    async def publish_event(self, event: AgentEvent) -> None:
        try:
            await self._redis.publish(self.channel, event.to_json())
            await self._redis.rpush(f"neocortex:session:{event.session_id}:events", event.to_json())
        except Exception:
            # Observability must never take the actual pipeline down with it.
            # A dashboard event being dropped is far better than a user-facing 500.
            pass

    async def subscribe(self) -> AsyncIterator[str]:
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self.channel)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield message["data"]
        finally:
            await pubsub.unsubscribe(self.channel)

    async def save_session_state(self, session_id: str, state: dict[str, Any], ttl_seconds: int = 3600) -> None:
        await self._redis.set(f"neocortex:session:{session_id}:state", json.dumps(state), ex=ttl_seconds)

    async def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        raw = await self._redis.get(f"neocortex:session:{session_id}:state")
        return json.loads(raw) if raw else None

    async def close(self) -> None:
        await self._redis.aclose()


class NullEventBus:
    """No-op bus used in tests / when Redis isn't available, so callers never need a None-check."""

    async def publish_event(self, event: AgentEvent) -> None:
        return None

    async def subscribe(self):
        return
        yield  # pragma: no cover - makes this an async generator

    async def save_session_state(self, *args, **kwargs) -> None:
        return None

    async def get_session_state(self, *args, **kwargs):
        return None

    async def close(self) -> None:
        return None
