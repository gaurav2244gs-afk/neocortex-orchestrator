"""
FastAPI app.

  POST /api/sessions      run a query through the full agent graph, return the final state
  GET  /api/sessions/{id} fetch a previously completed session's state from Redis
  GET  /api/health        liveness check
  WS   /ws/dashboard       live stream of every agent-graph event, for the dashboard
  GET  /dashboard          the dashboard itself (static HTML/JS, single file)

Everything that touches Redis/Chroma/the LLM is built once at startup and
stashed on `app.state`, so concurrent requests share connections/models
instead of re-initializing per-request -- this is what lets the API handle
many concurrent sessions without falling over.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from neocortex.agents.auditor import AuditorAgent
from neocortex.agents.executor import CorrectiveExecutorAgent, ExecutorAgent
from neocortex.agents.supervisor import SupervisorAgent
from neocortex.api.websocket_manager import ConnectionManager, redis_listener
from neocortex.config import settings
from neocortex.drift.semantic_drift import SemanticDriftDetector
from neocortex.embeddings import get_embedding_provider
from neocortex.graph.orchestrator import Agents, build_graph, run_session
from neocortex.graph.state import to_public_dict
from neocortex.llm.client import get_llm_client
from neocortex.memory.knowledge_graph import KnowledgeGraph
from neocortex.memory.redis_bus import EventBus, NullEventBus
from neocortex.nli.consistency import ConsistencyScorer

logger = logging.getLogger(__name__)


class SessionRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = get_embedding_provider(settings.embedding_model_name)
    knowledge_graph = KnowledgeGraph(
        embedding_provider=embedder,
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
    )
    consistency_scorer = ConsistencyScorer(settings.nli_model_name)
    drift_detector = SemanticDriftDetector(embedder)
    llm_client = get_llm_client(settings)

    try:
        event_bus: EventBus | NullEventBus = EventBus(settings.redis_url, settings.events_channel)
        await event_bus._redis.ping()  # fail fast if Redis isn't reachable
    except Exception as exc:
        logger.warning("Redis unavailable (%s); running with an in-process no-op event bus. "
                        "The dashboard will not receive live events.", exc)
        event_bus = NullEventBus()

    agents = Agents(
        executor=ExecutorAgent(llm_client),
        auditor=AuditorAgent(
            knowledge_graph=knowledge_graph,
            consistency_scorer=consistency_scorer,
            drift_detector=drift_detector,
            alpha=settings.nli_audit_weight,
            drift_threshold=settings.drift_threshold,
            confidence_floor=settings.confidence_floor,
        ),
        supervisor=SupervisorAgent(confidence_threshold=settings.confidence_threshold),
        corrective=CorrectiveExecutorAgent(llm_client),
    )
    compiled_graph = build_graph(agents, event_bus)

    manager = ConnectionManager()
    listener_task = asyncio.create_task(redis_listener(event_bus, manager))

    app.state.event_bus = event_bus
    app.state.compiled_graph = compiled_graph
    app.state.connection_manager = manager
    app.state.knowledge_graph = knowledge_graph

    yield

    listener_task.cancel()
    await event_bus.close()


app = FastAPI(title="NeoCortex Orchestrator", lifespan=lifespan)

_DASHBOARD_PATH = Path(__file__).parent / "dashboard" / "index.html"


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/sessions")
async def create_session(request: SessionRequest):
    session_id = request.session_id or str(uuid.uuid4())
    final_state = await run_session(
        compiled_graph=app.state.compiled_graph,
        event_bus=app.state.event_bus,
        session_id=session_id,
        query=request.query,
        max_retries=settings.max_retries,
        recursion_limit=settings.recursion_limit,
    )
    return to_public_dict(final_state)


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    state = await app.state.event_bus.get_session_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found (or it expired)")
    return to_public_dict(state)


@app.websocket("/ws/dashboard")
async def dashboard_socket(websocket: WebSocket):
    manager: ConnectionManager = app.state.connection_manager
    await manager.connect(websocket)
    try:
        while True:
            # We don't expect inbound messages, but reading keeps the connection
            # alive and lets us detect a client disconnect promptly.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/dashboard")
async def dashboard():
    return FileResponse(_DASHBOARD_PATH)
