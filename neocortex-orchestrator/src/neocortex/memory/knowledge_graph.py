"""
Ground-truth knowledge store.

This combines two pieces of the original stack on purpose:

- ChromaDB does the heavy lifting for similarity search over facts at
  scale (this is the "ChromaDB" in the stack).
- A small NetworkX graph sits on top to hold explicit subject->object
  relations between facts, so this is a genuine (if small) knowledge
  *graph* rather than just a flat vector index -- and it's what the
  Auditor and the semantic-drift detector both check responses against.

Embeddings are computed once via the shared `EmbeddingProvider` and handed
to Chroma explicitly (`query_embeddings=`) instead of letting Chroma pick
its own embedding function, so the embedding space used for drift
detection and the one used for retrieval are guaranteed to match.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx

from neocortex.embeddings import EmbeddingProvider, cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class Fact:
    id: str
    text: str
    subject: str | None = None
    relation: str | None = None
    obj: str | None = None
    metadata: dict = field(default_factory=dict)


class KnowledgeGraph:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        persist_dir: str = "./data/chroma",
        collection_name: str = "neocortex_facts",
    ):
        self.embedder = embedding_provider
        self.graph = nx.DiGraph()
        self._collection = None
        self._persist_dir = persist_dir
        self._collection_name = collection_name

    def _ensure_collection(self):
        if self._collection is not None:
            return
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = client.get_or_create_collection(
                name=self._collection_name, metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ChromaDB unavailable (%s); knowledge graph will use an in-memory "
                "linear scan instead of a persisted vector index.",
                exc,
            )
            self._collection = False  # sentinel: "tried and failed"

    def add_fact(self, fact: Fact) -> None:
        embedding = self.embedder.embed(fact.text)
        self.graph.add_node(fact.id, text=fact.text, embedding=embedding, metadata=fact.metadata)
        if fact.subject and fact.obj:
            self.graph.add_edge(fact.subject, fact.obj, relation=fact.relation, fact_id=fact.id)

        self._ensure_collection()
        if self._collection:
            self._collection.add(
                ids=[fact.id],
                documents=[fact.text],
                embeddings=[embedding.tolist()],
                metadatas=[fact.metadata or {}],
            )

    def add_facts(self, facts: list[Fact]) -> None:
        for fact in facts:
            self.add_fact(fact)

    def nearest_facts(self, text: str, k: int = 4) -> list[str]:
        """Top-k most similar fact texts, used as NLI premises and drift references."""
        query_embedding = self.embedder.embed(text)

        self._ensure_collection()
        if self._collection:
            result = self._collection.query(query_embeddings=[query_embedding.tolist()], n_results=k)
            docs = result.get("documents", [[]])[0]
            if docs:
                return docs

        # Fallback: linear scan over the in-memory graph (fine at this scale,
        # and keeps the project usable with zero external services).
        scored = []
        for _, data in self.graph.nodes(data=True):
            if "embedding" not in data:
                continue
            sim = cosine_similarity(query_embedding, data["embedding"])
            scored.append((sim, data["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:k]]

    def related_facts(self, entity: str) -> list[str]:
        """Facts explicitly linked to `entity` via the relation graph (true KG traversal)."""
        related = []
        for _, target, data in self.graph.out_edges(entity, data=True):
            fact_id = data.get("fact_id")
            if fact_id and fact_id in self.graph.nodes:
                related.append(self.graph.nodes[fact_id]["text"])
        return related

    def __len__(self) -> int:
        return sum(1 for _, d in self.graph.nodes(data=True) if "embedding" in d)
