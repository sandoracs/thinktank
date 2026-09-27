"""SQLiteMemoryBackend tests (DESIGN.md §11, §17).

Covers the production memory path: ``memory_items`` + ``memory_vec`` (vec0)
+ ``memory_fts`` (FTS5) + Reciprocal Rank Fusion, with the deterministic
:class:`FakeEmbeddingProvider` so the suite needs no model download.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from thinktank.domain.models import Layer
from thinktank.memory.embeddings import FakeEmbeddingProvider
from thinktank.memory.sqlite import SQLiteMemoryBackend, fts_query
from thinktank.storage.db import init_db, init_memory_tables, make_engine, stored_embedding_model


class NamedFake(FakeEmbeddingProvider):
    """A deterministic embedder with a configurable name (simulates a model swap)."""

    def __init__(self, dim: int, name: str) -> None:
        super().__init__(dim=dim)
        self.name = name

DIM = 16
SID1 = uuid.uuid4()
SID2 = uuid.uuid4()


@pytest.fixture
async def backend(tmp_path: Path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'mem.db'}", load_vec=True)
    await init_db(engine)
    embedder = FakeEmbeddingProvider(dim=DIM)
    await init_memory_tables(engine, DIM, embedder.name)
    yield SQLiteMemoryBackend(engine, embedder)
    await engine.dispose()


@pytest.mark.asyncio
async def test_add_and_search_roundtrip(backend: SQLiteMemoryBackend) -> None:
    item_id = await backend.add(
        "agent_a",
        Layer.LONG_TERM,
        "The evidence base for co-authorship claims is weak.",
        None,
        None,
        {"kind": "lesson"},
    )
    assert item_id >= 1

    hits = await backend.search("agent_a", "evidence co-authorship", 5, {Layer.LONG_TERM})
    assert len(hits) == 1
    assert hits[0].id == item_id
    assert hits[0].layer is Layer.LONG_TERM
    assert "evidence" in hits[0].content
    assert hits[0].score > 0


@pytest.mark.asyncio
async def test_search_is_scoped_to_agent(backend: SQLiteMemoryBackend) -> None:
    await backend.add("agent_a", Layer.LONG_TERM, "shared memory about models", None, None, {})
    await backend.add("agent_b", Layer.LONG_TERM, "shared memory about models", None, None, {})

    assert len(await backend.search("agent_a", "models", 5, {Layer.LONG_TERM})) == 1
    assert len(await backend.search("agent_b", "models", 5, {Layer.LONG_TERM})) == 1
    assert await backend.search("agent_c", "models", 5, {Layer.LONG_TERM}) == []


@pytest.mark.asyncio
async def test_layer_filter(backend: SQLiteMemoryBackend) -> None:
    await backend.add("agent_a", Layer.EPISODIC, "episodic note", SID1, None, {})
    await backend.add("agent_a", Layer.LONG_TERM, "long term note", None, None, {})

    long_hits = await backend.search("agent_a", "note", 5, {Layer.LONG_TERM})
    assert long_hits
    assert all(h.layer is Layer.LONG_TERM for h in long_hits)

    # Session-scoped search includes this session's episodic item plus
    # cross-session (NULL session) items, but never other sessions'.
    scoped = await backend.search(
        "agent_a", "note", 5, {Layer.EPISODIC, Layer.LONG_TERM}, session_id=SID1
    )
    assert {h.session_id for h in scoped} <= {SID1, None}


@pytest.mark.asyncio
async def test_other_sessions_episodic_items_are_excluded(backend: SQLiteMemoryBackend) -> None:
    await backend.add("agent_a", Layer.EPISODIC, "session one summary", SID1, None, {})
    hits = await backend.search(
        "agent_a", "summary", 5, {Layer.EPISODIC}, session_id=SID2
    )
    assert hits == []


@pytest.mark.asyncio
async def test_fts_ranks_term_matches_first(backend: SQLiteMemoryBackend) -> None:
    # FakeEmbedding is hash-based (no semantics), so vector ranking alone
    # cannot prefer the matching item. The FTS leg must carry it.
    await backend.add("agent_a", Layer.LONG_TERM, "irrelevant weather in berlin", None, None, {})
    await backend.add("agent_a", Layer.LONG_TERM, "the disclosure question", None, None, {})
    hits = await backend.search("agent_a", "disclosure", 5, {Layer.LONG_TERM})
    assert hits
    assert hits[0].content == "the disclosure question"


@pytest.mark.asyncio
async def test_dim_change_rebuilds_tables(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'dim.db'}", load_vec=True)
    await init_db(engine)
    embedder = FakeEmbeddingProvider(dim=DIM)
    await init_memory_tables(engine, DIM, embedder.name)
    backend = SQLiteMemoryBackend(engine, embedder)

    await backend.add("agent_a", Layer.LONG_TERM, "original item", None, None, {})
    assert await backend.search("agent_a", "original", 5, {Layer.LONG_TERM})

    # Swapping to a wider embedder drops the old memory and starts fresh.
    await init_memory_tables(engine, 32, "wider")
    assert await backend.search("agent_a", "original", 5, {Layer.LONG_TERM}) == []
    await engine.dispose()


@pytest.mark.asyncio
async def test_reembed_same_dim_keeps_items_and_updates_model(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 're.db'}", load_vec=True)
    await init_db(engine)
    old = FakeEmbeddingProvider(dim=DIM)
    await init_memory_tables(engine, DIM, old.name)
    backend = SQLiteMemoryBackend(engine, old)
    await backend.add("agent_a", Layer.LONG_TERM, "lesson about model ethics", None, None, {})
    assert await backend.search("agent_a", "ethics", 5, {Layer.LONG_TERM})

    # Same-dimension model swap: re-embed keeps the items (no fresh start).
    new = NamedFake(dim=DIM, name="new-model")
    count = await backend.reembed(new)
    assert count == 1
    assert await stored_embedding_model(engine) == "new-model"
    assert await backend.search("agent_a", "ethics", 5, {Layer.LONG_TERM})
    await engine.dispose()


@pytest.mark.asyncio
async def test_reembed_dim_change_rebuilds_vector_table(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 're2.db'}", load_vec=True)
    await init_db(engine)
    old = FakeEmbeddingProvider(dim=DIM)
    await init_memory_tables(engine, DIM, old.name)
    backend = SQLiteMemoryBackend(engine, old)
    await backend.add("agent_a", Layer.LONG_TERM, "cross dimension lesson", None, None, {})

    new = NamedFake(dim=32, name="wider")
    count = await backend.reembed(new)
    assert count == 1
    assert await stored_embedding_model(engine) == "wider"
    # A backend bound to the new provider can search the rebuilt table.
    wider = SQLiteMemoryBackend(engine, new)
    hits = await wider.search("agent_a", "lesson", 5, {Layer.LONG_TERM})
    assert len(hits) == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_reembed_empty_returns_zero(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 're3.db'}", load_vec=True)
    await init_db(engine)
    embedder = FakeEmbeddingProvider(dim=DIM)
    await init_memory_tables(engine, DIM, embedder.name)
    backend = SQLiteMemoryBackend(engine, embedder)
    assert await backend.reembed(embedder) == 0
    assert await stored_embedding_model(engine) == "fake"
    await engine.dispose()


def test_fts_query_escaping() -> None:
    assert fts_query("disclosure of AI usage") == '"disclosure" "of" "AI" "usage"'
    assert fts_query('quote "inside" word') == '"quote" "inside" "word"'
    assert fts_query("  ") == ""
    assert fts_query("a b") == ""  # tokens shorter than 2 chars are dropped
