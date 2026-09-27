"""Web hub tests (REST + WebSocket, FakeLLM, temp SQLite)."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from thinktank.config import Settings
from thinktank.llm.fake import FakeLLM
from thinktank.web.app import create_app


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    db = tmp_path / "web_test.db"
    # Hermetic: pin the test-critical fields (init kwargs outrank both OS env
    # vars and the developer's .env) so the suite never touches the network.
    settings = Settings(_env_file=None, fake_llm=True, embedding_backend="fake", data_dir=tmp_path)  # pyright: ignore[reportCallIssue]
    app = create_app(
        llm=FakeLLM(responses=[f"fake-reply-{i}" for i in range(60)]),
        database_url=f"sqlite+aiosqlite:///{db}",
        human_timeout_s=30,
        settings=settings,
    )
    with TestClient(app) as test_client:
        yield test_client
