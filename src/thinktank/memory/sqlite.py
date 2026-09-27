"""SQLite memory backend (DESIGN.md §7, §11).

The production :class:`MemoryBackend`: items live in ``memory_items``
(source of truth for the text), their vectors in the ``memory_vec`` vec0
table (``rowid`` = item id) and their full-text index in ``memory_fts``.
Search is hybrid — vector top-k + FTS5 top-k, fused with Reciprocal Rank
Fusion — and always scoped to one agent (DESIGN.md §11: memory is
per-agent, never shared).

The engine must have been created with ``load_vec=True`` and
``init_memory_tables(engine, dim)`` must have run for the embedder's
dimension; otherwise the vector side degrades to the FTS ranking only.
"""

from __future__ import annotations

import re
import struct
import uuid
from collections.abc import Sequence

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from thinktank.domain.models import Layer, MemoryHit
from thinktank.memory.base import MemoryBackend, reciprocal_rank_fusion
from thinktank.memory.embeddings import EmbeddingProvider
from thinktank.storage.tables import MemoryItem

#: Over-fetch factor for each index before the agent/layer filter, so the
#: fused result still has ``k`` survivors.
_OVERFETCH = 3

_TOKEN = re.compile(r"[\w\-]+", re.UNICODE)


def fts_query(query: str) -> str:
    """Build a safe FTS5 MATCH expression from free text (quoted terms, AND)."""
    terms = [t for t in _TOKEN.findall(query) if len(t) >= 2]
    if not terms:
        return ""
    return " ".join(f'"{t.replace("\"", "\"\"")}"' for t in terms)


def _pack_vector(vector: Sequence[float]) -> bytes:
    """Pack a float vector as the little-endian float32 BLOB sqlite-vec expects."""
    return struct.pack(f"<{len(vector)}f", *vector)


class SQLiteMemoryBackend(MemoryBackend):
    """Vec + FTS hybrid memory backend over SQLite (DESIGN.md §11)."""

    def __init__(self, engine: AsyncEngine, embedder: EmbeddingProvider) -> None:
        self._engine = engine
        self._embedder = embedder
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    async def add(
        self,
        agent_id: str,
        layer: Layer,
        content: str,
        session_id: uuid.UUID | None,
        source_seq: int | None,
        meta: dict[str, object],
    ) -> int:
        vector = (await self._embedder.embed([content]))[0]
        item = MemoryItem(
            agent_id=agent_id,
            layer=layer.value,
            content=content,
            session_id=str(session_id) if session_id is not None else None,
            source_seq=source_seq,
            meta=meta,
        )
        async with self._session_factory() as session:
            session.add(item)
            await session.flush()
            item_id = int(item.id)
            await session.execute(
                text("INSERT INTO memory_vec(rowid, embedding) VALUES(:id, :vec)"),
                {"id": item_id, "vec": _pack_vector(vector)},
            )
            await session.execute(
                text("INSERT INTO memory_fts(rowid, content) VALUES(:id, :content)"),
                {"id": item_id, "content": content},
            )
            await session.commit()
        return item_id

    async def search(
        self,
        agent_id: str,
        query: str,
        k: int,
        layers: set[Layer],
        session_id: uuid.UUID | None = None,
    ) -> list[MemoryHit]:
        async with self._session_factory() as session:
            candidates = await self._candidate_ids(session, agent_id, layers, session_id)
            if not candidates:
                return []
            candidate_set = set(candidates)

            vec_ranking = await self._vector_ranking(session, query, k, candidate_set)
            fts_ranking = await self._fts_ranking(session, query, k, candidate_set)
            scores = reciprocal_rank_fusion(vec_ranking, fts_ranking)
            if not scores:
                return []
            top = sorted(scores, key=lambda item_id: scores[item_id], reverse=True)[:k]

            rows = (
                await session.execute(
                    select(MemoryItem).where(MemoryItem.id.in_(top), MemoryItem.agent_id == agent_id)
                )
            ).scalars().all()
            by_id = {row.id: row for row in rows}

        hits: list[MemoryHit] = []
        for item_id in top:
            row = by_id.get(item_id)
            if row is None:
                continue
            hits.append(
                MemoryHit(
                    id=int(item_id),
                    layer=Layer(row.layer),
                    content=row.content,
                    score=scores[item_id],
                    session_id=uuid.UUID(row.session_id) if row.session_id else None,
                    source_seq=row.source_seq,
                    created_at=row.created_at,
                )
            )
        return hits

    async def reembed(self, embedder: EmbeddingProvider) -> int:
        """Re-embed every stored item with ``embedder`` (DESIGN.md §11, M3).

        Re-vectors all ``memory_items`` with the new provider and repairs any
        missing FTS rows. If the provider's dimension differs from the one the
        vector table was created for, the vector table is rebuilt. ``app_meta``
        is updated to record the new producing model and dimension. Returns the
        number of items re-embedded (0 when there is no memory).
        """
        new_dim = embedder.dim
        async with self._engine.begin() as conn:
            stored_dim = (
                await conn.exec_driver_sql("SELECT value FROM app_meta WHERE key = 'embedding_dim'")
            ).first()
            if stored_dim is not None and int(stored_dim[0]) != new_dim:
                await conn.exec_driver_sql("DROP TABLE IF EXISTS memory_vec;")
            await conn.exec_driver_sql(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(embedding float[{new_dim}]);"
            )
            rows = (await conn.exec_driver_sql("SELECT id, content FROM memory_items")).all()
            have_fts = {int(r[0]) for r in (await conn.exec_driver_sql("SELECT rowid FROM memory_fts")).all()}

        items = [(int(item_id), content) for item_id, content in rows]
        batch = 64
        for start in range(0, len(items), batch):
            chunk = items[start : start + batch]
            vectors = await embedder.embed([content for _item_id, content in chunk])
            async with self._engine.begin() as conn:
                for (item_id, content), vector in zip(chunk, vectors, strict=True):
                    await conn.exec_driver_sql("DELETE FROM memory_vec WHERE rowid = :id;", {"id": item_id})
                    await conn.exec_driver_sql(
                        "INSERT INTO memory_vec(rowid, embedding) VALUES(:id, :vec);",
                        {"id": item_id, "vec": _pack_vector(vector)},
                    )
                    if item_id not in have_fts:
                        await conn.exec_driver_sql(
                            "INSERT INTO memory_fts(rowid, content) VALUES(:id, :content);",
                            {"id": item_id, "content": content},
                        )
                await conn.exec_driver_sql(
                    "INSERT INTO app_meta(key, value) VALUES('embedding_dim', :v) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value;",
                    {"v": str(new_dim)},
                )
                await conn.exec_driver_sql(
                    "INSERT INTO app_meta(key, value) VALUES('embedding_model', :v) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value;",
                    {"v": embedder.name},
                )
        return len(items)

    # -- internals -----------------------------------------------------------
    async def _candidate_ids(
        self,
        session: AsyncSession,
        agent_id: str,
        layers: set[Layer],
        session_id: uuid.UUID | None,
    ) -> list[int]:
        stmt = select(MemoryItem.id).where(
            MemoryItem.agent_id == agent_id,
            MemoryItem.layer.in_([layer.value for layer in layers]),
        )
        if session_id is not None:
            # Long-term items have session_id NULL; episodic items belong to a
            # session. Include this session's items plus the cross-session ones.
            sid = str(session_id)
            stmt = stmt.where((MemoryItem.session_id == sid) | (MemoryItem.session_id.is_(None)))
        rows = (await session.execute(stmt)).scalars().all()
        return [int(r) for r in rows]

    async def _vector_ranking(
        self, session: AsyncSession, query: str, k: int, candidate_set: set[int]
    ) -> list[int]:
        try:
            vector = (await self._embedder.embed([query]))[0]
            limit = max(k, _OVERFETCH * k)
            rows = (
                await session.execute(
                    text(
                        "SELECT rowid FROM memory_vec WHERE embedding MATCH :q AND k = :k "
                        "ORDER BY distance"
                    ),
                    {"q": _pack_vector(vector), "k": limit},
                )
            ).all()
        except Exception:
            return []
        return [int(r[0]) for r in rows if int(r[0]) in candidate_set][:k]

    async def _fts_ranking(
        self, session: AsyncSession, query: str, k: int, candidate_set: set[int]
    ) -> list[int]:
        match = fts_query(query)
        if not match:
            return []
        try:
            limit = max(k, _OVERFETCH * k)
            rows = (
                await session.execute(
                    text(
                        "SELECT rowid FROM memory_fts WHERE memory_fts MATCH :q "
                        "ORDER BY bm25(memory_fts) LIMIT :limit"
                    ),
                    {"q": match, "limit": limit},
                )
            ).all()
        except Exception:
            return []
        return [int(r[0]) for r in rows if int(r[0]) in candidate_set][:k]
