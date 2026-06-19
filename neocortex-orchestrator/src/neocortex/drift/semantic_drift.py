"""
Semantic drift detection.

Drift is measured as 1 - (best cosine similarity between the current
response's embedding and a set of reference embeddings). The reference
set is the union of:
  - facts retrieved from the ground-truth knowledge graph for this query, and
  - the previous *accepted* response in this session, if any.

A drift score near 0 means the response stays close to something we can
verify or to where the conversation already was; a score near 1 means it
has wandered into territory unsupported by either.
"""
from __future__ import annotations

from neocortex.embeddings import EmbeddingProvider, cosine_similarity


class SemanticDriftDetector:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedder = embedding_provider

    def drift_score(self, current_text: str, reference_texts: list[str]) -> float:
        if not reference_texts:
            # Nothing to compare against yet (e.g. first turn, empty knowledge base).
            # Treat as neutral rather than claiming false confidence either way.
            return 0.5

        current_embedding = self.embedder.embed(current_text)
        reference_embeddings = self.embedder.embed_batch(reference_texts)
        best_similarity = max(
            cosine_similarity(current_embedding, ref) for ref in reference_embeddings
        )
        # Clamp to [0, 1]: cosine similarity can dip slightly negative for
        # unrelated text, which we still want to treat as "fully drifted".
        best_similarity = max(0.0, min(1.0, best_similarity))
        return 1.0 - best_similarity
