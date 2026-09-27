"""Embedding providers (DESIGN.md §11).

Two built-ins:
- :class:`FakeEmbeddingProvider` — deterministic hash vectors, no download.
  Used by tests and as a zero-setup default so the memory layer is exercisable
  without heavy models.
- :class:`LiteLLMEmbeddingProvider` — Ollama or API embeddings via LiteLLM.

A local :class:`sentence-transformers` provider (multilingual, recommended for
Hungarian) is intentionally an optional extra (``pip install thinktank[embeddings]``)
to keep the base install lean.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Embeds text batches into fixed-size vectors."""

    name: str
    dim: int

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free vectors (SHA-256 expanded to ``dim``).

    Not semantically meaningful, but stable across runs, which is exactly what
    the test-suite needs (retrieval order, RRF fusion, replay determinism).
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.name = "fake"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec: list[float] = []
        while len(vec) < self.dim:
            for byte in digest:
                vec.append((byte / 255.0) * 2.0 - 1.0)
                if len(vec) == self.dim:
                    break
        return [round(v, 6) for v in vec]


class LiteLLMEmbeddingProvider(EmbeddingProvider):
    """Embeddings through LiteLLM (Ollama or hosted API)."""

    def __init__(self, model: str, dim: int = 384) -> None:
        self.model = model
        self.dim = dim
        self.name = f"litellm:{model}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import litellm

        result = await litellm.aembedding(model=self.model, input=texts)
        data = result["data"]
        return [d["embedding"] for d in data]


def build_embedding_provider(*, backend: str, model: str, dim: int) -> EmbeddingProvider:
    """Construct the configured embedder (DESIGN.md §11).

    ``fake`` is the offline default (deterministic vectors, no download);
    ``litellm`` routes through Ollama or a hosted API.
    """
    if backend == "litellm":
        return LiteLLMEmbeddingProvider(model, dim=dim)
    return FakeEmbeddingProvider(dim=dim)
