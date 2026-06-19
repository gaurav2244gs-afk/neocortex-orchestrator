"""
LangGraph orchestrator.

Wires the 3-tier hierarchy into an actual graph:

    executor --> auditor --> supervisor --+--> [accept]           --> END
                    ^                     +--> [retry]            --> executor
                    |                     +--> [spawn_corrective] --> corrective --+
                    +-------------------------------------------------------------+
                                            +--> [fail]             --> END

Every node is wrapped with `_with_observability`, which times the call and
publishes an AgentEvent to the EventBus -- this is the one piece of plumbing
that makes the dashboard "see" agent cognition without each agent class
needing to know Redis exists.

NOTE: LangGraph's public API has moved fast across versions. This was
written against the `StateGraph` / `add_conditional_edges` / `END` API
common in 2024-2025 releases. Pin a known-good version in requirements.txt
and check the LangGraph docs if `graph.compile()` complains -- this is the
most likely spot to need a small adjustment after a library upgrade.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from langgraph.graph import END, StateGraph

from neocortex.agents.auditor import AuditorAgent
from neocortex.agents.executor import CorrectiveExecutorAgent, ExecutorAgent
from neocortex.agents.supervisor import SupervisorAgent, route_after_supervisor
from neocortex.graph.state import AgentState, initial_state
from neocortex.memory.redis_bus import AgentEvent, EventBus, NullEventBus


@dataclass
class Agents:
    executor: ExecutorAgent
    auditor: AuditorAgent
    supervisor: SupervisorAgent
    corrective: CorrectiveExecutorAgent


def _with_observability(node_name: str, fn, event_bus):
    async def wrapped(state: AgentState) -> AgentState:
        start = time.perf_counter()
        await event_bus.publish_event(AgentEvent(state["session_id"], node_name, "node_start"))

        result = fn(state)
        if hasattr(result, "__await__"):
            result = await result

        elapsed_ms = (time.perf_counter() - start) * 1000
        await event_bus.publish_event(
            AgentEvent(
                state["session_id"],
                node_name,
                "node_end",
                payload={
                    "elapsed_ms": round(elapsed_ms, 2),
                    "confidence_score": result.get("confidence_score"),
                    "drift_score": result.get("drift_score"),
                    "anomalies": result.get("anomalies"),
                    "status": result.get("status"),
                    "route": result.get("route"),
                    "token_usage": result.get("token_usage"),
                },
            )
        )
        return result

    return wrapped


def build_graph(agents: Agents, event_bus: EventBus | NullEventBus):
    graph = StateGraph(AgentState)

    graph.add_node("executor", _with_observability("executor", agents.executor.run, event_bus))
    graph.add_node("auditor", _with_observability("auditor", agents.auditor.run, event_bus))
    graph.add_node("supervisor", _with_observability("supervisor", agents.supervisor.run, event_bus))
    graph.add_node("corrective", _with_observability("corrective", agents.corrective.run, event_bus))

    graph.set_entry_point("executor")
    graph.add_edge("executor", "auditor")
    graph.add_edge("auditor", "supervisor")
    graph.add_edge("corrective", "auditor")

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "accept": END,
            "retry": "executor",
            "spawn_corrective": "corrective",
            "fail": END,
        },
    )

    return graph.compile()


async def run_session(
    compiled_graph,
    event_bus: EventBus | NullEventBus,
    session_id: str,
    query: str,
    max_retries: int,
    recursion_limit: int = 50,
) -> AgentState:
    state = initial_state(session_id=session_id, query=query, max_retries=max_retries)
    final_state = await compiled_graph.ainvoke(state, config={"recursion_limit": recursion_limit})
    await event_bus.publish_event(
        AgentEvent(session_id, "session", "session_complete", payload={"status": final_state.get("status")})
    )
    await event_bus.save_session_state(session_id, dict(final_state))
    return final_state
