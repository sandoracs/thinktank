"""LLM client abstraction.

The engine never calls a provider directly; it depends on the
:class:`LLMClient` protocol. That keeps the entire core testable with
:class:`thinktank.llm.fake.FakeLLM`, no API calls.

The concrete LiteLLM implementation (``litellm_client.py``) adds tenacity
retry, a concurrency semaphore, cost accounting and structured-output
fallback (native ``response_format`` first, then JSON extraction + one retry).
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

Purpose = Literal["speech", "summary", "reflection", "judge", "bid"]


class ChatMessage(BaseModel):
    """Provider-agnostic chat message (a subset of what LiteLLM accepts)."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResult(BaseModel):
    """Outcome of one completion call, rich enough to emit ``LLMCallCompleted``."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    parsed: BaseModel | None = None
    raw: object = None


@runtime_checkable
class LLMClient(Protocol):
    """Async LLM gateway. Implementations must be concurrency-safe."""

    async def complete(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        purpose: Purpose,
        temperature: float,
        max_tokens: int | None,
        response_model: type[BaseModel] | None = None,
    ) -> LLMResult:
        """Run one completion. ``response_model`` requests structured output."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (may be a no-op / cheap stub in fakes)."""
        ...
