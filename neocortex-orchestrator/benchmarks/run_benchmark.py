"""
Benchmark harness for NeoCortex Orchestrator.

This produces *measured* numbers from a real run on your machine -- it is
intentionally here so that any "reduced hallucination by X%" type claim in
your README or resume is something you can point at and reproduce, rather
than a number copied from somewhere else.

What it measures
-----------------
1. Hallucination rate, baseline vs. NeoCortex
   "Baseline" = a single Executor call with no auditing/correction loop
   (i.e. a vanilla LLM chain). "NeoCortex" = the full executor -> auditor
   -> supervisor graph. Both are scored against `eval_set.json`'s
   `facts` field using the same NLI consistency scorer used in
   production, since we don't have independently human-labeled ground
   truth here -- this is a proxy metric, not an external benchmark, and
   the script says so in its output.
2. Latency: mean / p95 per session.
3. Concurrency: N sessions run concurrently via asyncio.gather, reporting
   throughput. Defaults to the MockLLMClient so this measures orchestration
   overhead, not OpenAI's API latency -- rerun against a real backend
   (set USE_MOCK_LLM=false) for a production-representative number, and
   expect real-API concurrency to be bounded by your OpenAI rate limits
   rather than this code.

Usage
-----
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --concurrency 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
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
from neocortex.graph.state import initial_state  # noqa: E402
from neocortex.llm.client import MockLLMClient, get_llm_client  # noqa: E402
from neocortex.memory.knowledge_graph import Fact, KnowledgeGraph  # noqa: E402
from neocortex.memory.redis_bus import NullEventBus  # noqa: E402
from neocortex.nli.consistency import ConsistencyScorer  # noqa: E402

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
RESULTS_PATH = Path(__file__).parent / "results.json"
HALLUCINATION_SCORE_CUTOFF = 0.5  # NLI entailment below this counts as "hallucinated" for this benchmark


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text())


async def build_pipeline(use_mock: bool = True):
    embedder = get_embedding_provider(settings.embedding_model_name)
    kg = KnowledgeGraph(embedder, persist_dir="./data/benchmark_chroma", collection_name="benchmark_facts")
    for item in load_eval_set():
        for i, fact_text in enumerate(item["facts"]):
            kg.add_fact(Fact(id=f"{item['id']}-{i}", text=fact_text))

    scorer = ConsistencyScorer(settings.nli_model_name)
    drift = SemanticDriftDetector(embedder)
    llm = MockLLMClient(hallucination_rate=settings.mock_hallucination_rate) if use_mock else get_llm_client(settings)

    agents = Agents(
        executor=ExecutorAgent(llm),
        auditor=AuditorAgent(kg, scorer, drift, alpha=settings.nli_audit_weight),
        supervisor=SupervisorAgent(confidence_threshold=settings.confidence_threshold),
        corrective=CorrectiveExecutorAgent(llm),
    )
    bus = NullEventBus()
    graph = build_graph(agents, bus)
    return graph, bus, kg, scorer


async def run_baseline(executor: ExecutorAgent, query: str) -> str:
    state = initial_state(session_id="baseline", query=query, max_retries=0)
    result = await executor.run(state)
    return result["current_response"]


async def measure_hallucination_rate(graph, bus, kg, scorer, eval_set: list[dict]):
    baseline_executor = ExecutorAgent(MockLLMClient(hallucination_rate=settings.mock_hallucination_rate))

    baseline_hallucinations, neocortex_hallucinations = 0, 0
    latencies = []

    for item in eval_set:
        baseline_response = await run_baseline(baseline_executor, item["query"])
        baseline_score = scorer.score_against_context(baseline_response, item["facts"])
        if baseline_score < HALLUCINATION_SCORE_CUTOFF:
            baseline_hallucinations += 1

        start = time.perf_counter()
        final_state = await run_session(
            graph, bus, session_id=f"bench-{item['id']}", query=item["query"], max_retries=settings.max_retries
        )
        latencies.append(time.perf_counter() - start)

        final_score = scorer.score_against_context(final_state["current_response"], item["facts"])
        if final_score < HALLUCINATION_SCORE_CUTOFF:
            neocortex_hallucinations += 1

    n = len(eval_set)
    return {
        "n_queries": n,
        "baseline_hallucination_rate": baseline_hallucinations / n,
        "neocortex_hallucination_rate": neocortex_hallucinations / n,
        "relative_reduction_pct": (
            round(100 * (1 - (neocortex_hallucinations / max(baseline_hallucinations, 1))), 1)
            if baseline_hallucinations
            else None
        ),
        "mean_latency_s": round(statistics.mean(latencies), 4),
        "p95_latency_s": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 4),
    }


async def measure_concurrency(graph, bus, eval_set: list[dict], concurrency: int):
    queries = [eval_set[i % len(eval_set)]["query"] for i in range(concurrency)]

    async def one(i, q):
        start = time.perf_counter()
        await run_session(graph, bus, session_id=f"conc-{i}", query=q, max_retries=settings.max_retries)
        return time.perf_counter() - start

    start_all = time.perf_counter()
    latencies = await asyncio.gather(*(one(i, q) for i, q in enumerate(queries)))
    total_wall_time = time.perf_counter() - start_all

    return {
        "concurrent_sessions": concurrency,
        "total_wall_time_s": round(total_wall_time, 4),
        "throughput_sessions_per_s": round(concurrency / total_wall_time, 2),
        "mean_latency_s": round(statistics.mean(latencies), 4),
        "p95_latency_s": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 4),
        "note": "Measured against MockLLMClient (no external API calls) -- this isolates "
        "orchestration overhead from LLM-provider latency/rate limits. Rerun with "
        "USE_MOCK_LLM=false against a real backend for an end-to-end production number.",
    }


async def main(concurrency: int):
    eval_set = load_eval_set()
    graph, bus, kg, scorer = await build_pipeline(use_mock=True)

    print(f"Knowledge graph loaded with {len(kg)} facts.")
    print("Running hallucination-rate benchmark (baseline vs. NeoCortex self-correction loop)...")
    hallucination_results = await measure_hallucination_rate(graph, bus, kg, scorer, eval_set)

    print(f"Running concurrency benchmark ({concurrency} simulated concurrent sessions)...")
    concurrency_results = await measure_concurrency(graph, bus, eval_set, concurrency)

    results = {
        "hallucination_benchmark": hallucination_results,
        "concurrency_benchmark": concurrency_results,
        "disclaimer": (
            "These numbers are a proxy benchmark: hallucination is approximated via the "
            "same NLI model used in production, scored against a small hand-written fact "
            "set (see eval_set.json), not an independent human-labeled dataset. Treat as "
            "a reproducible sanity check, not a peer-reviewed evaluation. Re-run this "
            "script yourself before quoting any number publicly."
        ),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== Results ===")
    print(json.dumps(results, indent=2))
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=50, help="Number of simulated concurrent sessions")
    args = parser.parse_args()
    asyncio.run(main(args.concurrency))
