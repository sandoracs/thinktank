"""Settings page: configure everything from the .env file."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from thinktank.config import get_settings, reload_settings

ALL_FIELDS = [
    "default_model",
    "fake_llm",
    "llm_timeout_s",
    "llm_max_concurrency",
    "embedding_backend",
    "embedding_model",
    "embedding_dim",
    "host",
    "port",
    "database_url",
    "data_dir",
    "log_level",
    "log_json",
]

BASE_FORM = {
    "default_model": "ollama/qwen3.8",
    "llm_timeout_s": "90.5",
    "llm_max_concurrency": "4",
    "embedding_backend": "fake",
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "embedding_dim": "384",
    "host": "127.0.0.1",
    "port": "9099",
    "database_url": "sqlite+aiosqlite:///./thinktank.db",
    "data_dir": "data",
    "log_level": "DEBUG",
}


@pytest.fixture(autouse=True)
def _clean_thinktank_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings priority is OS env vars > ``.env`` file; drop any inherited
    ``THINKTANK_*`` vars so the file-based assertions stay deterministic."""
    for key in list(os.environ):
        if key.startswith("THINKTANK_"):
            monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()


def test_settings_page_lists_all_fields(client: TestClient) -> None:
    page = client.get("/settings")
    assert page.status_code == 200
    assert "Settings" in page.text
    for name in ALL_FIELDS:
        assert f'name="{name}"' in page.text


def test_settings_save_writes_env_and_applies(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    form = dict(BASE_FORM)
    form["fake_llm"] = "on"

    r = client.post("/settings", data=form, follow_redirects=False)
    assert r.status_code == 303, r.text
    assert r.headers["location"] == "/settings?saved=1"

    env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "THINKTANK_DEFAULT_MODEL=ollama/qwen3.8" in env
    assert "THINKTANK_FAKE_LLM=true" in env
    assert "THINKTANK_LLM_TIMEOUT_S=90.5" in env
    assert "THINKTANK_PORT=9099" in env

    # The reloaded in-process settings pick up the new values immediately.
    s = reload_settings()
    assert s.default_model == "ollama/qwen3.8"
    assert s.fake_llm is True
    assert s.llm_timeout_s == 90.5
    assert s.port == 9099
    assert s.log_json is False

    # The page now shows the stored values.
    page = client.get("/settings?saved=1")
    assert 'value="ollama/qwen3.8"' in page.text
    assert 'value="9099"' in page.text
    assert "Settings saved to" in page.text

    get_settings.cache_clear()  # let later tests re-read the real .env


def test_settings_save_rejects_invalid(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert client.post("/settings", data={**BASE_FORM, "llm_timeout_s": "abc"}).status_code == 400
    assert client.post("/settings", data={**BASE_FORM, "port": "99999"}).status_code == 400
    assert not (tmp_path / ".env").exists()
    get_settings.cache_clear()


def test_settings_save_and_restart(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import signal

    monkeypatch.chdir(tmp_path)
    calls: list[tuple[int, int]] = []

    def _kill(pid: int, sig: int) -> None:
        calls.append((pid, sig))

    monkeypatch.setattr(os, "kill", _kill)

    form = dict(BASE_FORM)
    form["fake_llm"] = "on"
    r = client.post("/settings/restart", data=form)
    assert r.status_code == 200, r.text
    assert "restarting" in r.text

    env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "THINKTANK_FAKE_LLM=true" in env
    assert "THINKTANK_PORT=9099" in env

    # The background task asks the process to exit so the start.sh loop restarts it.
    assert (os.getpid(), signal.SIGTERM) in calls
    get_settings.cache_clear()
