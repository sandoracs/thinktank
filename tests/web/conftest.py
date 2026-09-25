"""Web hub tests (DESIGN.md §17: REST + WebSocket, FakeLLM, temp SQLite)."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from roundtable.llm.fake import FakeLLM
from roundtable.web.app import create_app


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    db = tmp_path / "web_test.db"
    app = create_app(
        llm=FakeLLM(responses=[f"fake-reply-{i}" for i in range(60)]),
        database_url=f"sqlite+aiosqlite:///{db}",
        human_timeout_s=30,
    )
    with TestClient(app) as test_client:
        yield test_client
