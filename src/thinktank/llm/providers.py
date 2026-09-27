"""Registry of user-registered LLM providers (Settings → "LLM providers").

A registered provider is referenced by its ``id`` in model strings used by
agents or the default model: ``<id>/<model>`` is rewritten to the provider's
LiteLLM family with its ``api_base``/``api_key`` (OpenAI-compatible
endpoints, Azure, hosted Ollama, …). Providers without a matching ``id``
fall through untouched, so env-based setups (``OPENAI_API_KEY``,
``OLLAMA_API_BASE``) keep working.

Storage: ``<data_dir>/llm_providers.json`` (plain JSON, editable by hand).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# (LiteLLM provider family, Hungarian label)
PROVIDER_TYPES: list[tuple[str, str]] = [
    ("openai", "OpenAI / OpenAI-kompatibilis"),
    ("azure", "Azure OpenAI"),
    ("anthropic", "Anthropic"),
    ("gemini", "Google Gemini"),
    ("bedrock", "AWS Bedrock"),
    ("openrouter", "OpenRouter"),
    ("mistral", "Mistral"),
    ("groq", "Groq"),
    ("ollama", "Ollama (self-hosted)"),
]
PROVIDER_FAMILIES = {fam for fam, _ in PROVIDER_TYPES}
# Families with a universal default endpoint (azure/bedrock are tenant/region
# specific, so they are intentionally omitted and stay blank).
DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "openrouter": "https://openrouter.ai/api/v1",
    "mistral": "https://api.mistral.ai",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434",
}



class LLMProvider(BaseModel):
    """One registered provider endpoint."""

    id: str = Field(
        pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$",
        description="Short slug; models reference it as <id>/<model>.",
    )
    provider: str = Field(default="openai", description="LiteLLM provider family.")
    base_url: str | None = Field(default=None, description="Endpoint base URL (optional).")
    api_key: str | None = Field(default=None, description="API key (optional).")
    models: list[str] = Field(
        default_factory=list,
        description="Known model names offered by this endpoint (agent form dropdown).",
    )

    def key_masked(self) -> str | None:
        """Display form of the key (never the full value)."""
        if not self.api_key:
            return None
        if len(self.api_key) <= 8:
            return "•••"
        return f"{self.api_key[:3]}…{self.api_key[-4:]}"


class ProviderRegistry:
    """JSON-file backed list of :class:`LLMProvider` (small, read per call)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def list(self) -> list[LLMProvider]:
        if not self.path.is_file():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8") or "[]")
        return [LLMProvider(**item) for item in raw]

    def upsert(self, provider: LLMProvider) -> None:
        items = [p for p in self.list() if p.id != provider.id]
        items.append(provider)
        self._write(items)

    def remove(self, provider_id: str) -> bool:
        items = self.list()
        rest = [p for p in items if p.id != provider_id]
        if len(rest) == len(items):
            return False
        self._write(rest)
        return True

    def _write(self, items: list[LLMProvider]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([p.model_dump() for p in items], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def resolve(self, model: str) -> dict[str, Any]:
        """Map ``<id>/<model>`` to the registered provider.

        Returns ``{"model": ..., "api_base": ..., "api_key": ...}`` for a
        registered ``id``, else ``{}`` (model passes through unchanged).
        """
        head, sep, tail = model.partition("/")
        if not sep or not tail:
            return {}
        for p in self.list():
            if p.id == head:
                out: dict[str, Any] = {"model": f"{p.provider}/{tail}"}
                if p.base_url:
                    out["api_base"] = p.base_url
                if p.api_key:
                    out["api_key"] = p.api_key
                return out
        return {}

    def model_options(self) -> list[str]:
        """Every registered ``<id>/<model>`` pair, in registration order."""
        return [f"{p.id}/{m}" for p in self.list() for m in p.models]


def default_registry_path() -> Path:
    """``<data_dir>/llm_providers.json`` from the current settings."""
    from thinktank.config import get_settings

    return Path(get_settings().data_dir) / "llm_providers.json"
