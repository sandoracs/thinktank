"""LiteLLM-backed :class:`LLMClient`.

Responsibilities:
- tenacity retry on rate-limit / 5xx / connection / timeout errors;
- an ``asyncio.Semaphore`` bounding global concurrency;
- cost accounting via ``litellm.completion_cost``;
- structured output: request native JSON mode, then validate with Pydantic and
  retry once with the validation error fed back (for models without
  ``response_format`` support).

LiteLLM is imported lazily so the module (and the fake) load without it.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from pydantic import BaseModel

from thinktank.config import get_settings
from thinktank.llm.client import ChatMessage, LLMResult, Purpose
from thinktank.llm.providers import ProviderRegistry, default_registry_path


def _is_retryable(exc: BaseException) -> bool:
    """True for transient provider/network failures worth retrying."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    try:
        import litellm
    except Exception:
        return False
    candidates = (
        getattr(litellm, "RateLimitError", ()),
        getattr(litellm, "ServiceUnavailableError", ()),
        getattr(litellm, "InternalServerError", ()),
        getattr(litellm, "APIConnectionError", ()),
        getattr(litellm, "Timeout", ()),
    )
    types = tuple(t for t in candidates if t)
    return bool(types) and isinstance(exc, types)


def _extract_json(text: str) -> str:
    """Strip markdown code fences and trailing prose around a JSON blob."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


class LiteLLMClient:
    """Production LLM gateway (async, concurrency-bounded, cost-aware)."""

    def __init__(
        self,
        *,
        max_concurrency: int | None = None,
        timeout_s: float | None = None,
        providers: ProviderRegistry | None = None,
    ) -> None:
        settings = get_settings()
        self._semaphore = asyncio.Semaphore(max_concurrency or settings.llm_max_concurrency)
        self._timeout = timeout_s or settings.llm_timeout_s
        self._providers = providers or ProviderRegistry(default_registry_path())
        try:
            import litellm
        except ImportError:
            litellm = None
        if litellm is not None:
            # Some models reject non-default parameters (e.g. certain Claude
            # builds only accept temperature=1): drop them silently instead of
            # failing the whole turn.
            litellm.drop_params = True

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
        wire = [m.model_dump() for m in messages]
        async with self._semaphore:
            if response_model is not None:
                return await self._structured(model, wire, temperature, max_tokens, response_model)
            response = await self._acompletion(
                model=model,
                messages=wire,
                temperature=temperature,
                **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            )
            return self._to_result(response, model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import litellm

        settings = get_settings()
        result = await litellm.aembedding(model=settings.embedding_model, input=texts)
        return [d["embedding"] for d in result["data"]]

    async def _structured(
        self,
        model: str,
        wire: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        schema: type[BaseModel],
    ) -> LLMResult:

        wire = [*wire, {"role": "system", "content": "Respond with a single JSON object only."}]
        start = time.monotonic()
        response = await self._acompletion(
            model=model,
            messages=wire,
            temperature=temperature,
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        last_error: str | None = None
        for _ in range(2):
            try:
                schema.model_validate_json(_extract_json(content))
                break
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                wire = [
                    *wire,
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Invalid JSON: {last_error}. Fix and reply with JSON only."},
                ]
                response = await self._acompletion(
                    model=model,
                    messages=wire,
                    temperature=0.0,
                    **({"max_tokens": max_tokens} if max_tokens is not None else {}),
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
        duration_ms = int((time.monotonic() - start) * 1000)
        parsed: BaseModel | None = None
        try:
            parsed = schema.model_validate_json(_extract_json(content))
        except (ValueError, json.JSONDecodeError):
            parsed = None
        return self._to_result(response, model, content=content, parsed=parsed, duration_ms=duration_ms)

    async def _acompletion(self, **kwargs: Any) -> Any:
        import litellm
        import tenacity

        model = str(kwargs.get("model", ""))
        overrides = self._providers.resolve(model)
        if overrides:
            kwargs.update(overrides)
            model = overrides["model"]
        if model.startswith("ollama/") and "extra_body" not in kwargs:
            # Ollama thinking models spend the token budget on reasoning that
            # thinktank never keeps, returning empty content.
            kwargs["extra_body"] = {"think": False}

        @tenacity.retry(
            wait=tenacity.wait_exponential(multiplier=0.5, min=0.5, max=8),
            stop=tenacity.stop_after_attempt(4),
            retry=tenacity.retry_if_exception(_is_retryable),
            reraise=True,
        )
        async def _call() -> Any:
            async with asyncio.timeout(self._timeout):
                return await litellm.acompletion(**kwargs)

        return await _call()

    def _to_result(
        self,
        response: Any,
        model: str,
        *,
        content: str | None = None,
        parsed: BaseModel | None = None,
        duration_ms: int | None = None,
    ) -> LLMResult:
        import litellm

        text: str = content if content is not None else (response.choices[0].message.content or "")
        try:
            cost = litellm.completion_cost(completion_response=response) or 0.0
        except Exception:
            cost = 0.0
        usage = getattr(response, "usage", None)
        return LLMResult(
            content=text,
            model=getattr(response, "model", model) or model,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cost_usd=float(cost or 0.0),
            duration_ms=duration_ms if duration_ms is not None else 0,
            parsed=parsed,
            raw=response,
        )
