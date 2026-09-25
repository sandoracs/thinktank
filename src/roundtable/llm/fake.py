"""Deterministic fake LLM for the test-suite and offline development.

Two modes:
- Scripted: feed an ordered list of responses (plain text, or JSON strings for
  structured-output calls); each ``complete()`` pops the next one.
- Responder: supply a ``responder(model, messages, purpose, response_model)``
  callable to produce context-dependent answers (used by integration tests that
  assert on *how* the agent behaved, not just *that* it did).

Token counts are derived from word counts so cost-limit and budget code paths
are exercised with meaningful, deterministic numbers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from roundtable.llm.client import ChatMessage, LLMResult, Purpose
from roundtable.memory.embeddings import EmbeddingProvider, FakeEmbeddingProvider

Responder = Callable[[str, list[ChatMessage], Purpose, type[BaseModel] | None], str]


def _words(text: str) -> int:
    return len(text.split())


#: A reusable pool of plausible debate turns for offline FakeLLM runs.
SAMPLE_TURNS: list[str] = [
    "I think we should start from the premise that the question is real, not hypothetical.",
    "With respect, that assumes the evidence is as strong as it looks.",
    "Could you give one concrete example where that has actually held up?",
    "I largely agree, though I'd push back on the strength of the causal claim.",
    "That's a fair point; let me steelman the other side before responding.",
    "Here is where I diverge: the incentives, not the intentions, are what matter.",
    "I'm not convinced that resolves the disclosure question you raised earlier.",
    "If we accept that, then the practical standard becomes much simpler.",
    "Let me be specific about the risk I see in your framing.",
    "That's the strongest version of the argument so far, but it leaves a gap.",
    "I'd like to concede one point before restating my core position.",
    "To summarise where we land: we agree on the goal but disagree on the rule.",
]


def sample_responses(n: int = 60) -> list[str]:
    """Deterministic response pool large enough for a multi-round session."""
    base = SAMPLE_TURNS * (n // len(SAMPLE_TURNS) + 1)
    return base[:n]


class FakeLLM:
    """An :class:`LLMClient` that never touches the network."""

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        responder: Responder | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._responder = responder
        self._embedder = embedder or FakeEmbeddingProvider()
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        purpose: Purpose,
        temperature: float,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
    ) -> LLMResult:
        content = self._next(model, messages, purpose, response_model)
        parsed: BaseModel | None = None
        if response_model is not None and content.strip():
            try:
                parsed = response_model.model_validate_json(content)
            except (json.JSONDecodeError, ValueError):
                parsed = None

        input_tokens = sum(_words(m.content) for m in messages)
        output_tokens = _words(content)
        result = LLMResult(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,
            duration_ms=1,
            parsed=parsed,
        )
        self.calls.append(
            {
                "model": model,
                "purpose": purpose,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": list(messages),
                "response_model": response_model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )
        return result

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._embedder.embed(texts)

    def _next(
        self,
        model: str,
        messages: list[ChatMessage],
        purpose: Purpose,
        response_model: type[BaseModel] | None,
    ) -> str:
        if self._responder is not None:
            return self._responder(model, messages, purpose, response_model)
        if self._responses:
            return self._responses.pop(0)
        return f"[fake:{purpose}]"
