"""LLM provider registration on the Settings page."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from thinktank.llm.litellm_client import LiteLLMClient
from thinktank.llm.providers import LLMProvider, ProviderRegistry


def test_provider_page_starts_empty(client: TestClient) -> None:
    page = client.get("/settings")
    assert page.status_code == 200
    assert "LLM providers" in page.text
    assert "No providers registered yet" in page.text
    # Default base URLs ship with the page (auto-fill for known families).
    assert "https://api.openai.com/v1" in page.text
    assert "https://api.mistral.ai" in page.text


def test_provider_add_list_mask_delete(client: TestClient, tmp_path: Path) -> None:
    r = client.post(
        "/settings/providers",
        data={
            "id": "myopenai",
            "provider": "openai",
            "base_url": "https://api.example.com/v1",
            "api_key": "sk-secret-key-1234567890",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text

    page = client.get("/settings")
    assert "<code>myopenai</code>" in page.text
    assert "https://api.example.com/v1" in page.text
    # The key is masked, never shown in full.
    assert "sk-secret-key-1234567890" not in page.text
    assert "sk-…7890" in page.text

    # Persisted under the app's data_dir.
    registry = ProviderRegistry(tmp_path / "llm_providers.json")
    stored = registry.list()
    assert [p.id for p in stored] == ["myopenai"]
    assert stored[0].api_key == "sk-secret-key-1234567890"

    # Upsert on the same id, then delete.
    r2 = client.post(
        "/settings/providers",
        data={"id": "myopenai", "provider": "azure", "base_url": "", "api_key": ""},
        follow_redirects=False,
    )
    assert r2.status_code == 303, r2.text
    assert [p.id for p in registry.list()] == ["myopenai"]
    assert registry.list()[0].provider == "azure"

    rd = client.post("/settings/providers/delete", data={"id": "myopenai"}, follow_redirects=False)
    assert rd.status_code == 303
    assert registry.list() == []


def test_provider_edit_link_prefills_and_overwrites(client: TestClient, tmp_path: Path) -> None:
    """Edit is a link that server-prefills the form; saving overwrites the row."""
    r = client.post(
        "/settings/providers",
        data={
            "id": "prov1",
            "provider": "openai",
            "base_url": "https://api.example.com/v1",
            "api_key": "sk-keep-1234567890",
            "models": "gpt-4o",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text

    # The row offers an Edit link for the provider.
    page = client.get("/settings").text
    assert 'href="/settings?edit=prov1"' in page
    assert 'readonly' not in page

    # The edit page prefills every field, locks the ID, and announces the mode.
    edit_page = client.get("/settings?edit=prov1").text
    assert 'value="prov1"' in edit_page
    assert "readonly" in edit_page
    assert 'value="https://api.example.com/v1"' in edit_page
    assert 'value="gpt-4o"' in edit_page
    assert '<option value="openai" selected>' in edit_page
    assert 'Editing "prov1"' in edit_page
    assert "Save overwrites this row" in edit_page

    # An unknown edit id falls back to the create form.
    ghost = client.get("/settings?edit=ghost").text
    assert "readonly" not in ghost
    assert 'Editing "' not in ghost

    # Saving the prefilled form overwrites the row (one entry, not a new one).
    r2 = client.post(
        "/settings/providers",
        data={
            "id": "prov1",
            "provider": "openai",
            "base_url": "https://api.example.com/v1",
            "models": "gpt-4o, gpt-4o-mini",
        },
        follow_redirects=False,
    )
    assert r2.status_code == 303, r2.text
    stored = ProviderRegistry(tmp_path / "llm_providers.json").list()
    assert [p.id for p in stored] == ["prov1"]
    assert stored[0].api_key == "sk-keep-1234567890"
    assert stored[0].models == ["gpt-4o", "gpt-4o-mini"]


    # An explicit key replaces the stored one.
    r3 = client.post(
        "/settings/providers",
        data={"id": "prov1", "provider": "openai", "api_key": "sk-new-9876543210"},
        follow_redirects=False,
    )
    assert r3.status_code == 303, r3.text
    assert ProviderRegistry(tmp_path / "llm_providers.json").list()[0].api_key == "sk-new-9876543210"


def test_provider_models_listing(client: TestClient, tmp_path: Path) -> None:
    """/api/provider-models lists the models an endpoint actually serves."""
    import http.server
    import json
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.endswith("/api/tags"):
                payload = {"models": [{"name": "stub-ollama"}]}
            elif self.path.endswith("/models"):
                payload = {"data": [{"id": "stub-b"}, {"id": "stub-a"}]}
            else:
                payload = {}
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        # OpenAI-compatible listing.
        r = client.post("/api/provider-models", json={"provider": "openai", "base_url": base})
        assert r.status_code == 200, r.text
        assert r.json()["models"] == ["stub-a", "stub-b"]

        # Ollama listing.
        r2 = client.post("/api/provider-models", json={"provider": "ollama", "base_url": base})
        assert r2.json()["models"] == ["stub-ollama"]

        # Registered-provider mode.
        client.post(
            "/settings/providers",
            data={"id": "stubprov", "provider": "openai", "base_url": base},
            follow_redirects=False,
        )
        r3 = client.post("/api/provider-models", json={"id": "stubprov"})
        assert r3.json()["models"] == ["stub-a", "stub-b"]

        # Unknown id -> 404; unreachable endpoint -> 502 with a message.
        assert client.post("/api/provider-models", json={"id": "ghost"}).status_code == 404
        bad = client.post("/api/provider-models", json={"provider": "openai", "base_url": "http://127.0.0.1:1"})
        assert bad.status_code == 502
        assert "unavailable" in bad.json()["detail"]

        # Azure/Bedrock cannot be listed -> 502 with guidance.
        azure = client.post("/api/provider-models", json={"provider": "azure"})
        assert azure.status_code == 502
        assert "Other" in azure.json()["detail"]
    finally:
        server.shutdown()
        server.server_close()


def test_provider_models_feed_agent_form(client: TestClient, tmp_path: Path) -> None:
    """Registered models become the only options of the agent form dropdown."""
    r = client.post(
        "/settings/providers",
        data={
            "id": "myprov",
            "provider": "openai",
            "base_url": "https://api.example.com/v1",
            "api_key": "sk-test-1234567890",
            "models": "gpt-4o, gpt-4o-mini",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text

    # The settings table lists the models of the provider.
    assert "gpt-4o, gpt-4o-mini" in client.get("/settings").text

    # New-agent form: a select offering exactly the registered models.
    form = client.get("/agents/new").text
    assert '<select name="model"' in form
    assert '<option value="myprov/gpt-4o"' in form
    assert '<option value="myprov/gpt-4o-mini"' in form

    # A form submission picking a registered model is stored.
    post = client.post(
        "/agents",
        data={
            "id": "dropdown_agent",
            "model": "myprov/gpt-4o-mini",
            "temperature": "0.7",
            "persona_name": "D",
            "persona_role": "r",
            "drift_mode": "bounded",
            "consistency_threshold": "3",
        },
        follow_redirects=False,
    )
    assert post.status_code == 303, post.text
    stored = {a["id"]: a for a in client.get("/api/agents").json()}
    assert stored["dropdown_agent"]["model"] == "myprov/gpt-4o-mini"

    # Editing an agent whose model is not registered keeps it selectable.
    client.post(
        "/api/agents",
        json={
            "id": "legacy_model",
            "model": "ollama/qwen3.8",
            "persona": {"name": "L", "role": "r"},
        },
    )
    edit = client.get("/agents/legacy_model/edit").text
    assert '<option value="ollama/qwen3.8" selected>ollama/qwen3.8 (current)</option>' in edit


def test_agent_form_empty_when_no_models(client: TestClient) -> None:
    """Without registered models the dropdown is disabled with a hint."""
    form = client.get("/agents/new").text
    assert '<select name="model" disabled' in form
    assert "Settings" in form


def test_provider_validation(client: TestClient, tmp_path: Path) -> None:
    # Uppercase id is normalized to lowercase; genuinely bad ids (space) -> 400.
    r = client.post(
        "/settings/providers", data={"id": "Owentest", "provider": "openai"}, follow_redirects=False
    )
    assert r.status_code == 303, r.text
    assert [p.id for p in ProviderRegistry(tmp_path / "llm_providers.json").list()] == ["owentest"]
    assert client.post("/settings/providers", data={"id": "Bad ID", "provider": "openai"}).status_code == 400
    # Bad base URL -> 400.
    assert (
        client.post(
            "/settings/providers", data={"id": "ok", "provider": "openai", "base_url": "not-a-url"}
        ).status_code
        == 400
    )
    # Unknown provider family -> 400.
    assert client.post("/settings/providers", data={"id": "ok", "provider": "telepathy"}).status_code == 400
    # Deleting a missing provider -> 404.
    assert client.post("/settings/providers/delete", data={"id": "ghost"}).status_code == 404


def test_resolve_registered_vs_unknown(tmp_path: Path) -> None:
    registry = ProviderRegistry(tmp_path / "llm_providers.json")
    registry.upsert(
        LLMProvider(id="myopenai", provider="openai", base_url="https://api.example.com/v1", api_key="sk-abc")
    )
    got = registry.resolve("myopenai/gpt-4o")
    assert got == {
        "model": "openai/gpt-4o",
        "api_base": "https://api.example.com/v1",
        "api_key": "sk-abc",
    }
    # Unknown ids and plain model names pass through.
    assert registry.resolve("ollama/qwen3.8") == {}
    assert registry.resolve("gpt-4o") == {}


def test_client_passes_provider_creds_to_litellm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    from typing import Any

    import litellm

    registry = ProviderRegistry(tmp_path / "llm_providers.json")
    registry.upsert(
        LLMProvider(id="myopenai", provider="openai", base_url="https://api.example.com/v1", api_key="sk-abc")
    )
    client = LiteLLMClient(providers=registry, timeout_s=5)

    calls: list[dict[str, Any]] = []

    def _resp() -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
            model="openai/gpt-4o",
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
        )

    async def fake_acompletion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _resp()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    def _cost(**kwargs: Any) -> float:
        return 0.0

    monkeypatch.setattr(litellm, "completion_cost", _cost)

    asyncio.run(
        client.complete(
            model="myopenai/gpt-4o",
            messages=[],
            purpose="speech",
            temperature=0.7,
            max_tokens=100,
        )
    )
    assert calls[0]["model"] == "openai/gpt-4o"
    assert calls[0]["api_base"] == "https://api.example.com/v1"
    assert calls[0]["api_key"] == "sk-abc"


def test_client_enables_drop_params(tmp_path: Path) -> None:
    """Models that reject params (e.g. temperature!=1) get the params dropped, not an error."""
    import litellm

    assert litellm.drop_params in (True, False)
    LiteLLMClient(providers=ProviderRegistry(tmp_path / "llm_providers.json"))
    assert litellm.drop_params is True
