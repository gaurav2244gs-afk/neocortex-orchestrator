"""
Seeds the persistent knowledge graph (./data/chroma by default) with a
small starter set of facts, so a freshly cloned repo has something for
the Auditor to check responses against. In a real deployment you'd point
this at your actual domain knowledge base instead.

Run with:
    python examples/seed_knowledge_base.py
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from neocortex.config import settings  # noqa: E402
from neocortex.embeddings import get_embedding_provider  # noqa: E402
from neocortex.memory.knowledge_graph import Fact, KnowledgeGraph  # noqa: E402

STARTER_FACTS = [
    Fact(id="eiffel-tower", text="The Eiffel Tower was completed in 1889 and is located in Paris, France."),
    Fact(id="mount-everest", text="Mount Everest is approximately 8,849 meters tall."),
    Fact(id="water-boiling", text="Water boils at 100 degrees Celsius at sea level."),
    Fact(id="python-creator", text="The Python programming language was created by Guido van Rossum, first released in 1991."),
    Fact(id="moon-landing", text="Humans first landed on the Moon in 1969, during the Apollo 11 mission."),
    Fact(
        id="great-wall",
        text="The Great Wall of China was built primarily to protect Chinese states from invasions and raids.",
        subject="great-wall-of-china",
        relation="built_for",
        obj="border-defense",
    ),
]


def main():
    embedder = get_embedding_provider(settings.embedding_model_name)
    kg = KnowledgeGraph(embedder, persist_dir=settings.chroma_persist_dir, collection_name=settings.chroma_collection)
    kg.add_facts(STARTER_FACTS)
    print(f"Seeded knowledge graph with {len(kg)} facts at '{settings.chroma_persist_dir}'.")


if __name__ == "__main__":
    main()
