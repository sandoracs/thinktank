"""Memory backend abstraction.

The three layers (working / episodic / long-term) are addressed by
:class:`thinktank.domain.models.Layer`. Working memory is not stored here — it
is projected from the message stream — so the backend only persists episodic
and long-term items.

Retrieval is hybrid: vector top-k + FTS top-k, fused with Reciprocal Rank
Fusion. :func:`reciprocal_rank_fusion` is a pure function so it can be unit
tested in isolation.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from thinktank.domain.models import Layer, MemoryConfig, MemoryHit
from thinktank.memory.embeddings import EmbeddingProvider, FakeEmbeddingProvider

#: Rank constant for Reciprocal Rank Fusion (standard k=60).
RRF_K = 60


def reciprocal_rank_fusion(
    *rankings: Sequence[int],
    k: int = RRF_K,
) -> dict[int, float]:
    """Fuse multiple ranked id lists into a single score map.

    ``score(id) = Σ_rankings 1 / (k + rank_in_ranking)`` where ``rank`` is
    zero-based. Ids absent from a ranking simply contribute nothing from it.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class MemoryBackend(ABC):
    """Persists and retrieves per-agent memory items."""

    @abstractmethod
    async def add(
        self,
        agent_id: str,
        layer: Layer,
        content: str,
        session_id: uuid.UUID | None,
        source_seq: int | None,
        meta: dict[str, object],
    ) -> int:
        """Store an item and return its opaque id."""
        ...

    @abstractmethod
    async def search(
        self,
        agent_id: str,
        query: str,
        k: int,
        layers: set[Layer],
        session_id: uuid.UUID | None = None,
    ) -> list[MemoryHit]:
        """Return up to ``k`` items for ``agent_id`` matching ``query``."""
        ...


class InMemoryMemoryBackend(MemoryBackend):
    """List-backed backend with vector-similarity search.

    Used by the test-suite and as a zero-setup default. It is NOT the
    production backend (that is :class:`SQLiteMemoryBackend`, M3), but it
    satisfies the same contract so the engine and AIAgent code paths are
    identical.
    """

    def __init__(self, embedder: EmbeddingProvider | None = None) -> None:
        self._embedder = embedder or FakeEmbeddingProvider()
        self._next_id = 1
        self._items: list[dict[str, Any]] = []

    async def add(
        self,
        agent_id: str,
        layer: Layer,
        content: str,
        session_id: uuid.UUID | None,
        source_seq: int | None,
        meta: dict[str, object],
    ) -> int:
        embedding = (await self._embedder.embed([content]))[0]
        item: dict[str, Any] = {
            "id": self._next_id,
            "agent_id": agent_id,
            "layer": layer,
            "content": content,
            "session_id": session_id,
            "source_seq": source_seq,
            "meta": meta,
            "embedding": embedding,
        }
        self._items.append(item)
        self._next_id += 1
        return int(item["id"])

    async def search(
        self,
        agent_id: str,
        query: str,
        k: int,
        layers: set[Layer],
        session_id: uuid.UUID | None = None,
    ) -> list[MemoryHit]:
        candidates = [
            it
            for it in self._items
            if it["agent_id"] == agent_id and it["layer"] in layers
            and (session_id is None or it["session_id"] in (session_id, None))
        ]
        if not candidates:
            return []
        query_vec = (await self._embedder.embed([query]))[0]
        scored = sorted(
            candidates,
            key=lambda it: cosine_similarity(query_vec, it["embedding"]),
            reverse=True,
        )
        hits: list[MemoryHit] = []
        for it in scored[:k]:
            hits.append(
                MemoryHit(
                    id=int(it["id"]),
                    layer=it["layer"],
                    content=str(it["content"]),
                    score=cosine_similarity(query_vec, it["embedding"]),
                    session_id=it["session_id"],
                    source_seq=it["source_seq"],
                )
            )
        return hits


def default_memory_config() -> MemoryConfig:
    return MemoryConfig()
