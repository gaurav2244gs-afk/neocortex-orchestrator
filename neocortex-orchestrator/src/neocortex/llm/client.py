"""
Pluggable LLM client.

`OpenAIChatClient` talks to GPT-4 (or whatever model you configure) via the
official OpenAI SDK. `MockLLMClient` is a deterministic, dependency-free
stand-in used when no API key is configured -- it lets the rest of the
pipeline (graph wiring, auditing, drift detection, dashboard) run and be
demoed/tested without spending real API credits.

Swap in another provider (Anthropic, local vLLM, etc.) by implementing the
same `LLMClient` protocol.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(Protocol):
    async def generate(self, messages: list[dict[str, str]]) -> LLMResponse: ...


class OpenAIChatClient:
    """Real GPT-4 backend via the OpenAI SDK (`pip install openai`)."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        from openai import AsyncOpenAI  # imported lazily so MockLLMClient never needs the package

        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate(self, messages: list[dict[str, str]]) -> LLMResponse:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
        )
        choice = resp.choices[0].message.content or ""
        usage = resp.usage
        return LLMResponse(
            text=choice,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class MockLLMClient:
    """
    Deterministic offline stand-in for an LLM.

    It echoes back a plausible-looking answer derived from the prompt, and
    -- with probability `hallucination_rate` -- deliberately drops or
    contradicts a fact so that the auditor/drift-detection loop has
    something real to catch. This is what lets `benchmarks/run_benchmark.py`
    produce an actual before/after hallucination-rate number without an API
    key, instead of asking you to take a metric on faith.
    """

    _CONTRADICTION_BANK = [
        "Note: I'm not fully certain, but I believe this happened in a different year than usually cited.",
        "Actually, I recall this being located in a different city.",
        "I think the commonly cited figure for this is somewhat exaggerated.",
    ]

    def __init__(self, hallucination_rate: float = 0.35, seed: int | None = None):
        self.hallucination_rate = hallucination_rate
        self._rng = random.Random(seed)

    async def generate(self, messages: list[dict[str, str]]) -> LLMResponse:
        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        query = user_turns[-1] if user_turns else ""
        is_correction_pass = any(
            m["role"] == "system" and "revise" in m["content"].lower() for m in messages
        )

        base = f"Based on the available information: {query.strip().rstrip('?')}."

        if is_correction_pass:
            # Corrective passes are biased toward being faithful to the supplied context.
            grounded = next(
                (m["content"] for m in messages if m["role"] == "system" and "Verified facts" in m["content"]),
                None,
            )
            text = f"{base} " + (f"Revised answer grounded in the verified facts provided." if grounded else "Revised answer.")
        elif self._rng.random() < self.hallucination_rate:
            text = f"{base} {self._rng.choice(self._CONTRADICTION_BANK)}"
        else:
            text = f"{base} This is consistent with the established record."

        prompt_tokens = sum(len(m["content"].split()) for m in messages)
        completion_tokens = len(text.split())
        return LLMResponse(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def get_llm_client(settings) -> LLMClient:
    """Factory: real GPT-4 client if an API key is configured, mock otherwise."""
    if not settings.use_mock_llm and settings.openai_api_key:
        return OpenAIChatClient(api_key=settings.openai_api_key, model=settings.openai_model)
    return MockLLMClient(hallucination_rate=settings.mock_hallucination_rate)
