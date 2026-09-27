"""sqlite-vec under aiosqlite.

The design's top risk: loading the sqlite-vec extension into an aiosqlite
connection on macOS. This test is the regression guard for that gate — it
fails if the extension cannot be loaded (threading) or queried.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlite_vec

from thinktank.storage.db import init_db, make_engine


@pytest.mark.asyncio
async def test_sqlite_vec_loads_and_queries_under_aiosqlite(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'vec.db'}", load_vec=True)
    await init_db(engine)
    async with engine.connect() as conn:
        await conn.exec_driver_sql("CREATE VIRTUAL TABLE vec_test USING vec0(embedding float[4])")
        blob = sqlite_vec.serialize_float32([1.0, 2.0, 3.0, 4.0])
        await conn.exec_driver_sql("INSERT INTO vec_test(rowid, embedding) VALUES (1, ?)", (blob,))
        rows = (
            await conn.exec_driver_sql(
                "SELECT rowid, vec_distance_cosine(embedding, ?) FROM vec_test "
                "WHERE embedding MATCH ? AND k = 1 ORDER BY distance",
                (blob, blob),
            )
        ).fetchall()
        assert rows[0][0] == 1
        assert abs(rows[0][1]) < 1e-6  # exact vector -> cosine distance ~0
    await engine.dispose()


@pytest.mark.asyncio
async def test_wal_mode_is_set_on_real_database(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'wal.db'}")
    await init_db(engine)
    async with engine.connect() as conn:
        row = (await conn.exec_driver_sql("PRAGMA journal_mode;")).fetchone()
        assert row is not None
        mode = row[0]
    await engine.dispose()
    assert mode.lower() == "wal"
