"""
Shared embedding utilities.

Both the semantic-drift detector and the knowledge-graph store need text
embeddings in the *same* vector space, so we centralise that here rather
than letting Chroma and the drift module each pick their own model.

If `sentence-transformers` (and its model weights) aren't available --
e.g. no network access -- we fall back to a deterministic hashed
bag-of-words embedding. It is far weaker than a real sentence embedding,
but it keeps the whole pipeline runnable offline/in CI for demos and
tests, and it is swapped out transparently the moment a real model can
be loaded.
"""
from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

_FALLBACK_DIM = 256
_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def _hash_embed(text: str, dim: int = _FALLBACK_DIM) -> np.ndarray:
    """Deterministic, dependency-free pseudo-embedding (hashed bag-of-words)."""
    vec = np.zeros(dim, dtype=np.float32)
    for token in _TOKEN_RE.findall(text.lower()):
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class EmbeddingProvider:
    """Wraps a SentenceTransformer model with an offline-safe fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._tried_load = False

    def _ensure_model(self):
        if self._tried_load:
            return
        self._tried_load = True
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info("Loaded embedding model '%s'", self.model_name)
        except Exception as exc:  # noqa: BLE001 - genuinely want to catch any load failure
            logger.warning(
                "Could not load sentence-transformers model '%s' (%s). "
                "Falling back to hashed bag-of-words embeddings.",
                self.model_name,
                exc,
            )
            self._model = None

    def embed(self, text: str) -> np.ndarray:
        self._ensure_model()
        if self._model is not None:
            vec = self._model.encode(text, normalize_embeddings=True)
            return np.asarray(vec, dtype=np.float32)
        return _hash_embed(text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        self._ensure_model()
        if self._model is not None and texts:
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return [np.asarray(v, dtype=np.float32) for v in vecs]
        return [_hash_embed(t) for t in texts]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, robust to zero vectors. Returns a value in [-1, 1]."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@lru_cache(maxsize=4)
def get_embedding_provider(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingProvider:
    """Process-wide cached provider so the (possibly large) model loads once."""
    return EmbeddingProvider(model_name)
