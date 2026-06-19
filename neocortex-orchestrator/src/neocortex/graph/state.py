"""
Shared state passed between every node in the LangGraph agent graph.

Keeping this as a single TypedDict (rather than scattering state across
agent objects) is what lets LangGraph checkpoint, resume, and reason about
the graph -- and it's what we serialize straight into Redis for the
dashboard and for session resumption.
"""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    session_id: str
    query: str

    history: list[dict[str, str]]          # [{"role": "user"|"assistant", "content": ...}, ...]
    current_response: str
    context_chunks: list[str]                # facts retrieved from the knowledge graph this turn
    corrective_context: str | None           # feedback injected into the next executor call

    confidence_score: float
    drift_score: float
    anomalies: list[str]

    retry_count: int
    max_retries: int
    used_corrective_subagent: bool

    status: str      # "pending" | "correcting" | "accepted" | "failed"
    route: str        # set by the supervisor node: "accept" | "retry" | "spawn_corrective" | "fail"

    token_usage: dict[str, int]   # {"prompt_tokens": int, "completion_tokens": int}


def initial_state(session_id: str, query: str, max_retries: int) -> AgentState:
    return AgentState(
        session_id=session_id,
        query=query,
        history=[],
        current_response="",
        context_chunks=[],
        corrective_context=None,
        confidence_score=0.0,
        drift_score=0.0,
        anomalies=[],
        retry_count=0,
        max_retries=max_retries,
        used_corrective_subagent=False,
        status="pending",
        route="",
        token_usage={"prompt_tokens": 0, "completion_tokens": 0},
    )


def merge_token_usage(state: AgentState, prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    usage = dict(state.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0})
    usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + prompt_tokens
    usage["completion_tokens"] = usage.get("completion_tokens", 0) + completion_tokens
    return usage


def to_public_dict(state: AgentState) -> dict[str, Any]:
    """Trim internal routing fields before returning state to an API client."""
    return {
        "session_id": state.get("session_id"),
        "query": state.get("query"),
        "response": state.get("current_response"),
        "confidence_score": round(state.get("confidence_score", 0.0), 4),
        "drift_score": round(state.get("drift_score", 0.0), 4),
        "anomalies": state.get("anomalies", []),
        "retries": state.get("retry_count", 0),
        "used_corrective_subagent": state.get("used_corrective_subagent", False),
        "status": state.get("status"),
        "token_usage": state.get("token_usage"),
    }
