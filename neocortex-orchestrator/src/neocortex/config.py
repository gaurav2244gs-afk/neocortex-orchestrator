"""
Central configuration for NeoCortex Orchestrator.

All values are overridable via environment variables / a .env file.
See .env.example for the full list of knobs.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM backend -----------------------------------------------------
    openai_api_key: str | None = None
    openai_model: str = "gpt-4"
    use_mock_llm: bool = True  # safe default so the repo runs out of the box with no API key
    mock_hallucination_rate: float = 0.35  # used only by MockLLMClient, see llm/client.py

    # --- Redis (event bus / session state) --------------------------------
    redis_url: str = "redis://localhost:6379/0"
    events_channel: str = "neocortex:events"

    # --- ChromaDB (fact / knowledge-graph store) ---------------------------
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "neocortex_facts"

    # --- Embeddings & NLI ---------------------------------------------------
    embedding_model_name: str = "all-MiniLM-L6-v2"
    nli_model_name: str = "cross-encoder/nli-deberta-v3-base"

    # --- Self-correction loop thresholds ------------------------------------
    confidence_threshold: float = 0.75   # supervisor accepts a response above this
    drift_threshold: float = 0.40        # drift score above this is flagged as an anomaly
    confidence_floor: float = 0.20       # below this is flagged as an anomaly regardless of retries
    max_retries: int = 2                 # executor re-tries before escalating to corrective sub-agent
    nli_audit_weight: float = 0.6        # alpha: weight of NLI entailment vs (1 - drift) in confidence

    # --- API ------------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000

    # --- LangGraph -------------------------------------------------------------
    recursion_limit: int = 50


settings = Settings()
