"""
Tier 3: Supervisor agent.

Doesn't talk to an LLM at all -- it's pure routing logic over the state
the Auditor just produced. It is what implements "dynamically re-routes or
spawns corrective sub-agents when confidence drops below threshold":

  - confidence is high enough              -> accept, stop.
  - confidence is low but retries remain    -> send back to the Executor
                                                with concrete feedback.
  - retries exhausted, not yet escalated    -> spawn the corrective sub-agent
                                                (stricter, facts-only prompt).
  - corrective sub-agent also failed        -> give up, flag as failed.

This is implemented as a graph *node* (not just an edge-routing function)
because it needs to mutate state (retry_count, corrective_context,
status) before the conditional edge reads `state["route"]`.
"""
from __future__ import annotations

from neocortex.graph.state import AgentState


class SupervisorAgent:
    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold

    def run(self, state: AgentState) -> AgentState:
        new_state: AgentState = dict(state)  # type: ignore[assignment]
        confidence = state.get("confidence_score", 0.0)
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)

        if confidence >= self.confidence_threshold:
            new_state["status"] = "accepted"
            new_state["route"] = "accept"
            return new_state

        if retry_count < max_retries:
            new_state["retry_count"] = retry_count + 1
            new_state["corrective_context"] = self._build_feedback(state)
            new_state["status"] = "correcting"
            new_state["route"] = "retry"
            return new_state

        if not state.get("used_corrective_subagent", False):
            new_state["used_corrective_subagent"] = True
            new_state["corrective_context"] = self._build_feedback(state)
            new_state["status"] = "correcting"
            new_state["route"] = "spawn_corrective"
            return new_state

        new_state["status"] = "failed"
        new_state["route"] = "fail"
        return new_state

    @staticmethod
    def _build_feedback(state: AgentState) -> str:
        confidence = state.get("confidence_score", 0.0)
        drift = state.get("drift_score", 0.0)
        chunks = state.get("context_chunks", [])
        facts_block = "\n".join(f"- {c}" for c in chunks) if chunks else "(no supporting facts retrieved)"
        return (
            f"Your previous answer scored confidence={confidence:.2f}, drift={drift:.2f}, "
            f"which is below the acceptance threshold. Revise your answer so it is fully "
            f"consistent with these verified facts:\n{facts_block}"
        )


def route_after_supervisor(state: AgentState) -> str:
    """LangGraph conditional-edge function: read the route the supervisor node already decided."""
    return state.get("route", "fail")
