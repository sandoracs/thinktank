"""Async engine, pragmas, and schema bootstrap (DESIGN.md §7, §17).

Per the design, every connection applies ``journal_mode=WAL``,
``foreign_keys=ON``, and ``busy_timeout=5000``. The sqlite-vec extension is
loaded on connection only when the vector backend is active, avoiding the
load cost (and the aiosqlite threading edge) when it is not needed.

Virtual tables (``vec0`` / FTS5) are not ORM models; they are created here
alongside the ORM tables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from roundtable.config import get_settings
from roundtable.storage.tables import Base


def resolve_database_url(url: str | None = None) -> str:
    """Resolve a relative SQLite path against the configured data directory."""
    url = url or get_settings().database_url
    prefix = "sqlite+aiosqlite:///"
    if url.startswith(prefix):
        path = url[len(prefix):]
        if path and path != ":memory:" and not os.path.isabs(path):
            base = get_settings().data_dir
            resolved = base / path
            # SQLite can create the file but not its parent directories; make
            # sure the data directory exists so the first run just works.
            resolved.parent.mkdir(parents=True, exist_ok=True)
            return f"{prefix}{resolved}"
    return url


def vec_library_path() -> str:
    """Locate the bundled sqlite-vec shared library."""
    import sqlite_vec

    root = Path(sqlite_vec.__file__).parent
    for name in ("vec0.dylib", "vector0.dylib", "vec0.so"):
        candidate = root / name
        if candidate.exists():
            return str(candidate)
    msg = f"sqlite-vec library not found under {root}"
    raise RuntimeError(msg)


def make_engine(
    url: str | None = None,
    *,
    load_vec: bool = False,
    echo: bool = False,
) -> AsyncEngine:
    """Build an async engine with the required pragmas (and optional vec)."""
    engine = create_async_engine(resolve_database_url(url), echo=echo, future=True)

    def _on_connect(dbapi_conn: Any, _record: Any) -> None:
        # aiosqlite runs its sqlite3 connection on a background thread; the
        # SQLAlchemy adapter bridges that work via ``await_``. Pragmas run
        # through the adapter cursor (which already routes to the thread), and
        # the sqlite-vec load must happen in that same thread, so we drive the
        # aiosqlite connection's extension methods directly.
        aiosqlite_conn = getattr(dbapi_conn, "_connection", None)
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.execute("PRAGMA busy_timeout=5000;")
            if load_vec and aiosqlite_conn is not None:
                await_ = dbapi_conn.await_
                await_(aiosqlite_conn.enable_load_extension(True))
                await_(aiosqlite_conn.load_extension(vec_library_path()))
        finally:
            cursor.close()

    event.listens_for(engine.sync_engine, "connect")(_on_connect)
    return engine


FTS_TABLES = (
    (
        "messages_fts",
        "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts "
        "USING fts5(content, content='messages', content_rowid='rowid');",
    ),
    (
        "memory_fts",
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
        "USING fts5(content, content='memory_items', content_rowid='id');",
    ),
)


async def init_db(engine: AsyncEngine) -> None:
    """Create ORM tables and the FTS5 indexes. Idempotent."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with engine.begin() as conn:
        for _name, ddl in FTS_TABLES:
            await conn.exec_driver_sql(ddl)


async def _meta(conn: Any, key: str) -> str | None:
    row = (await conn.exec_driver_sql("SELECT value FROM app_meta WHERE key = :k", {"k": key})).first()
    return row[0] if row is not None else None


async def init_memory_tables(engine: AsyncEngine, dim: int, model: str | None = None) -> None:
    """Create the memory vector virtual table for ``dim``-dimensional vectors.

    The dimension is persisted in ``app_meta``. If the configured dimension
    changes, the old vectors are dropped and rebuilt and the (now orphaned)
    memory items are cleared — a dimension change is a fresh start. When the
    dimension is unchanged the existing vectors are kept, and
    ``embedding_model`` (the model that produced the current vectors) is only
    recorded when the tables are (re)created, so a later same-dimension model
    swap stays detectable for :func:`reembed` / the hub's startup check.
    """
    async with engine.begin() as conn:
        stored = await _meta(conn, "embedding_dim")
        dim_changed = stored is not None and int(stored) != dim
        if dim_changed:
            await conn.exec_driver_sql("DROP TABLE IF EXISTS memory_fts;")
            await conn.exec_driver_sql("DROP TABLE IF EXISTS memory_vec;")
            await conn.exec_driver_sql("DELETE FROM memory_items;")
            await conn.exec_driver_sql(
                "CREATE VIRTUAL TABLE memory_fts "
                "USING fts5(content, content='memory_items', content_rowid='id');"
            )
        await conn.exec_driver_sql(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(embedding float[{dim}]);"
        )
        await conn.exec_driver_sql(
            "INSERT INTO app_meta(key, value) VALUES('embedding_dim', :v) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value;",
            {"v": str(dim)},
        )
        if model is not None and (stored is None or dim_changed):
            await conn.exec_driver_sql(
                "INSERT INTO app_meta(key, value) VALUES('embedding_model', :v) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value;",
                {"v": model},
            )


async def stored_embedding_model(engine: AsyncEngine) -> str | None:
    """Return the model that produced the current memory vectors, if recorded.

    ``None`` when the memory layer has not been initialised. Compare against
    the configured provider's name to detect a model swap that needs
    ``roundtable reembed``.
    """
    async with engine.connect() as conn:
        return await _meta(conn, "embedding_model")


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker[AsyncSession](engine, expire_on_commit=False, autoflush=False)


async def dispose(engine: AsyncEngine) -> None:
    """Dispose the engine's connection pool (call at shutdown)."""
    await engine.dispose()
