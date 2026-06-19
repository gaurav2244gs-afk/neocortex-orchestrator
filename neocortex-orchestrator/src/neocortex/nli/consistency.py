"""
Factual consistency scoring for the Auditor agent.

`ConsistencyScorer` uses a cross-encoder NLI model to estimate the
probability that a retrieved fact (premise) entails the agent's response
(hypothesis). Averaged across the top-k retrieved facts, this becomes the
NLI half of the confidence score (see agents/auditor.py).

NOTE on label order: `cross-encoder/nli-deberta-v3-base` outputs logits in
the order [contradiction, entailment, neutral] per its model card. If you
swap in a different NLI checkpoint, verify its label order -- this is the
single most common source of silently-inverted confidence scores in NLI
pipelines, and we'd rather you catch it here than in production.

If `sentence-transformers`/the model weights aren't available (no network,
no GPU, whatever), we fall back to a lexical-overlap heuristic. It's a much
weaker signal, but it keeps the pipeline runnable end-to-end offline.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def _word_overlap_score(premise: str, hypothesis: str) -> float:
    p = set(_TOKEN_RE.findall(premise.lower()))
    h = set(_TOKEN_RE.findall(hypothesis.lower()))
    if not h:
        return 0.0
    return len(p & h) / len(h)


class ConsistencyScorer:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        self.model_name = model_name
        self._model = None
        self._tried_load = False

    def _ensure_model(self):
        if self._tried_load:
            return
        self._tried_load = True
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            logger.info("Loaded NLI model '%s'", self.model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load NLI model '%s' (%s). Falling back to lexical-overlap scoring.",
                self.model_name,
                exc,
            )
            self._model = None

    def score(self, premise: str, hypothesis: str) -> float:
        """Returns an entailment probability in [0, 1]."""
        self._ensure_model()
        if self._model is None:
            return _word_overlap_score(premise, hypothesis)

        import numpy as np

        logits = self._model.predict([(premise, hypothesis)])[0]
        probs = np.exp(logits) / np.exp(logits).sum()
        # label order for cross-encoder/nli-deberta-v3-base: [contradiction, entailment, neutral]
        entailment_prob = float(probs[1])
        return entailment_prob

    def score_against_context(self, response: str, context_chunks: list[str]) -> float:
        """Average entailment probability across all retrieved context chunks.

        Returns 0.5 (maximally uncertain) when there is no context to check
        against at all, rather than silently treating "nothing to compare
        against" as either high or low confidence.
        """
        if not context_chunks:
            return 0.5
        scores = [self.score(premise=chunk, hypothesis=response) for chunk in context_chunks]
        return sum(scores) / len(scores)
