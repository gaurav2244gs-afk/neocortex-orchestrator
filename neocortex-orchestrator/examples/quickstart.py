"""
Minimal example: run a single query through the agent graph directly,
without going through the FastAPI layer. Useful for understanding the
core library, or for embedding NeoCortex in another application.

Run with:
    python examples/quickstart.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from neocortex.agents.auditor import AuditorAgent  # noqa: E402
from neocortex.agents.executor import CorrectiveExecutorAgent, ExecutorAgent  # noqa: E402
from neocortex.agents.supervisor import SupervisorAgent  # noqa: E402
from neocortex.config import settings  # noqa: E402
from neocortex.drift.semantic_drift import SemanticDriftDetector  # noqa: E402
from neocortex.embeddings import get_embedding_provider  # noqa: E402
from neocortex.graph.orchestrator import Agents, build_graph, run_session  # noqa: E402
from neocortex.graph.state import to_public_dict  # noqa: E402
from neocortex.llm.client import get_llm_client  # noqa: E402
from neocortex.memory.knowledge_graph import Fact, KnowledgeGraph  # noqa: E402
from neocortex.memory.redis_bus import NullEventBus  # noqa: E402
from neocortex.nli.consistency import ConsistencyScorer  # noqa: E402


async def main():
    embedder = get_embedding_provider(settings.embedding_model_name)
    kg = KnowledgeGraph(embedder, persist_dir="./data/quickstart_chroma")
    kg.add_fact(Fact(id="fact-1", text="The Eiffel Tower was completed in 1889 and is located in Paris, France."))

    llm = get_llm_client(settings)  # MockLLMClient unless OPENAI_API_KEY + USE_MOCK_LLM=false are set
    agents = Agents(
        executor=ExecutorAgent(llm),
        auditor=AuditorAgent(kg, ConsistencyScorer(settings.nli_model_name), SemanticDriftDetector(embedder)),
        supervisor=SupervisorAgent(confidence_threshold=settings.confidence_threshold),
        corrective=CorrectiveExecutorAgent(llm),
    )
    graph = build_graph(agents, NullEventBus())

    final_state = await run_session(
        graph, NullEventBus(), session_id="quickstart-1",
        query="When was the Eiffel Tower completed?", max_retries=settings.max_retries,
    )

    print(to_public_dict(final_state))


if __name__ == "__main__":
    asyncio.run(main())
