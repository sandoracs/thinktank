"""Same-origin guard: cross-site POSTs/WebSockets and foreign Host headers are refused."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from thinktank.web.security import host_only, parse_allowed_hosts


def test_cross_site_form_post_is_refused(client: TestClient) -> None:
    r = client.post(
        "/settings/providers",
        data={"id": "evil", "provider": "openai", "base_url": "https://evil.example"},
        headers={"Origin": "https://evil.example"},
    )
    assert r.status_code == 403
    assert "evil" not in client.get("/settings").text


def test_sec_fetch_site_cross_site_is_refused(client: TestClient) -> None:
    r = client.post("/settings/restart", data={}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


def test_same_origin_post_passes_guard(client: TestClient) -> None:
    r = client.post(
        "/settings/providers/delete", data={"id": "nope"}, headers={"Origin": "http://localhost"}
    )
    assert r.status_code == 404  # reached the handler (no such provider)


def test_post_without_origin_passes_guard(client: TestClient) -> None:
    r = client.post("/settings/providers/delete", data={"id": "nope"})
    assert r.status_code == 404


def test_foreign_host_is_refused_even_for_get(client: TestClient) -> None:
    r = client.get("/", headers={"Host": "attacker.example"})
    assert r.status_code == 403


def test_cross_site_websocket_is_refused(client: TestClient) -> None:
    created = client.post("/api/sessions", json={"template": "default"})
    sid = created.json().get("id", "00000000-0000-0000-0000-000000000000") if created.status_code < 400 else (
        "00000000-0000-0000-0000-000000000000"
    )
    with pytest.raises(WebSocketDisconnect), client.websocket_connect(
        f"/ws/sessions/{sid}", headers={"Origin": "https://evil.example"}
    ) as ws:
        ws.receive_json()


def test_helpers() -> None:
    assert parse_allowed_hosts("*") is None
    assert parse_allowed_hosts("Localhost, [::1]") == frozenset({"localhost", "::1"})
    assert host_only("127.0.0.1:8080") == "127.0.0.1"
    assert host_only("[::1]:8080") == "::1"
    assert host_only("localhost") == "localhost"
