"""
Bridges the Redis event-bus pub/sub channel to any number of connected
dashboard WebSocket clients. One background task subscribes to Redis once;
every connected browser tab just gets fanned-out copies.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket

from neocortex.memory.redis_bus import EventBus, NullEventBus

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast(self, message: str) -> None:
        stale = []
        for connection in self._connections:
            try:
                await connection.send_text(message)
            except Exception:
                stale.append(connection)
        for connection in stale:
            self.disconnect(connection)


async def redis_listener(event_bus: EventBus | NullEventBus, manager: ConnectionManager) -> None:
    """Long-running background task: forward every bus event to all dashboard clients."""
    if isinstance(event_bus, NullEventBus):
        return
    try:
        async for message in event_bus.subscribe():
            await manager.broadcast(message)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Redis listener crashed; dashboard will stop receiving live events.")
