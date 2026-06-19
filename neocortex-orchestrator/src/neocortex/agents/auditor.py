"""
Tier 2: Auditor agent.

Scores the Executor's latest response on two independent axes and combines
them into a single confidence score the Supervisor can threshold against:

  1. Factual consistency -- NLI entailment probability between the
     response and facts retrieved from the ground-truth knowledge graph.
  2. Semantic drift -- how far the response's embedding has wandered from
     those same facts (and from the previous accepted response, if any).

confidence = alpha * nli_score + (1 - alpha) * (1 - drift_score)

Either component flags an anomaly on its own if it crosses a hard floor,
independent of whether the combined score happens to clear the threshold --
a single very-low component shouldn't be allowed to hide behind a healthy
average.
"""
from __future__ import annotations

from neocortex.drift.semantic_drift import SemanticDriftDetector
from neocortex.graph.state import AgentState
from neocortex.memory.knowledge_graph import KnowledgeGraph
from neocortex.nli.consistency import ConsistencyScorer


class AuditorAgent:
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        consistency_scorer: ConsistencyScorer,
        drift_detector: SemanticDriftDetector,
        alpha: float = 0.6,
        drift_threshold: float = 0.40,
        confidence_floor: float = 0.20,
        top_k_facts: int = 4,
    ):
        self.knowledge_graph = knowledge_graph
        self.consistency_scorer = consistency_scorer
        self.drift_detector = drift_detector
        self.alpha = alpha
        self.drift_threshold = drift_threshold
        self.confidence_floor = confidence_floor
        self.top_k_facts = top_k_facts

    async def run(self, state: AgentState) -> AgentState:
        response = state["current_response"]
        context_chunks = self.knowledge_graph.nearest_facts(state["query"], k=self.top_k_facts)

        reference_texts = list(context_chunks)
        previous_accepted = self._previous_accepted_response(state)
        if previous_accepted:
            reference_texts.append(previous_accepted)

        nli_score = self.consistency_scorer.score_against_context(response, context_chunks)
        drift = self.drift_detector.drift_score(response, reference_texts)
        confidence = self.alpha * nli_score + (1 - self.alpha) * (1 - drift)

        anomalies = list(state.get("anomalies", []))
        if drift > self.drift_threshold:
            anomalies.append(
                f"semantic_drift={drift:.2f} exceeds threshold={self.drift_threshold:.2f}"
            )
        if confidence < self.confidence_floor:
            anomalies.append(f"confidence={confidence:.2f} below hard floor={self.confidence_floor:.2f}")

        new_state: AgentState = dict(state)  # type: ignore[assignment]
        new_state["context_chunks"] = context_chunks
        new_state["confidence_score"] = confidence
        new_state["drift_score"] = drift
        new_state["anomalies"] = anomalies
        return new_state

    @staticmethod
    def _previous_accepted_response(state: AgentState) -> str | None:
        history = state.get("history", [])
        assistant_turns = [m["content"] for m in history if m["role"] == "assistant"]
        # The *current* response is always the last assistant turn appended by the
        # executor, so the previously accepted one (if any) is the one before that.
        return assistant_turns[-2] if len(assistant_turns) >= 2 else None
