"""
Tier 1: Executor agents.

`ExecutorAgent` produces the actual answer to the user's query. When the
Supervisor sends it back with `corrective_context`, that feedback is
injected as an extra system message so the next attempt is grounded in
whatever the Auditor flagged as inconsistent.

`CorrectiveExecutorAgent` is the "corrective sub-agent" the Supervisor can
spawn after ordinary retries are exhausted: same interface, but a much
stricter system prompt that forces it to answer only from the verified
facts it's handed, rather than free-generating.
"""
from __future__ import annotations

from neocortex.graph.state import AgentState, merge_token_usage
from neocortex.llm.client import LLMClient

_BASE_SYSTEM_PROMPT = (
    "You are a careful, factual assistant. Answer the user's question directly and concisely."
)

_CORRECTIVE_SYSTEM_PROMPT = (
    "You are a fact-checking specialist invoked after a previous answer was flagged as "
    "potentially inconsistent with known facts. Re-answer the user's question. Revise your "
    "previous answer so it is fully supported by the verified facts below, and do not include "
    "any claim that isn't supported by them. If the facts don't cover something, say so plainly "
    "instead of guessing."
)


class ExecutorAgent:
    system_prompt = _BASE_SYSTEM_PROMPT

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def _build_messages(self, state: AgentState) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(state.get("history", []))

        corrective_context = state.get("corrective_context")
        if corrective_context:
            messages.append({"role": "system", "content": corrective_context})

        messages.append({"role": "user", "content": state["query"]})
        return messages

    async def run(self, state: AgentState) -> AgentState:
        messages = self._build_messages(state)
        response = await self.llm_client.generate(messages)

        new_state: AgentState = dict(state)  # type: ignore[assignment]
        new_state["current_response"] = response.text
        new_state["token_usage"] = merge_token_usage(
            state, response.prompt_tokens, response.completion_tokens
        )
        history = list(state.get("history", []))
        history.append({"role": "assistant", "content": response.text})
        new_state["history"] = history
        new_state["status"] = "pending"
        return new_state


class CorrectiveExecutorAgent(ExecutorAgent):
    system_prompt = _CORRECTIVE_SYSTEM_PROMPT

    def _build_messages(self, state: AgentState) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        context_chunks = state.get("context_chunks", [])
        if context_chunks:
            facts_block = "\n".join(f"- {c}" for c in context_chunks)
            messages.append({"role": "system", "content": f"Verified facts:\n{facts_block}"})

        corrective_context = state.get("corrective_context")
        if corrective_context:
            messages.append({"role": "system", "content": corrective_context})

        messages.append({"role": "user", "content": state["query"]})
        return messages
