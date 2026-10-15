"""Same-origin guard for the hub.

The hub has no authentication; binding to loopback keeps other machines out,
but not *other websites open in the user's own browser*:

- **Cross-site requests.** A page on any origin can auto-submit a form to
  ``http://127.0.0.1:8080/settings/providers`` (register a provider whose base
  URL it controls, so later prompts go to it), ``/settings`` (rewrite ``.env``)
  or ``/settings/restart``. Form POSTs are "simple" requests, so CORS never
  blocks them. Likewise a cross-site page can open the session WebSocket and
  read every transcript.
- **DNS rebinding.** An attacker domain re-pointed to 127.0.0.1 becomes
  same-origin with the hub as far as the browser is concerned; only the
  ``Host`` header still shows the attacker's name.

This middleware therefore (1) rejects requests whose ``Host`` is not in the
allow-list, and (2) rejects state-changing HTTP requests and WebSocket
handshakes whose ``Origin`` (or ``Sec-Fetch-Site``) shows they came from
another site. Non-browser clients (curl, scripts, the CLI) send no ``Origin``
and are unaffected.
"""

from __future__ import annotations

import json
from urllib.parse import urlsplit

from starlette.types import ASGIApp, Message, Receive, Scope, Send

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
DEFAULT_ALLOWED_HOSTS = "localhost,127.0.0.1,::1"


def parse_allowed_hosts(value: str) -> frozenset[str] | None:
    """Comma list -> lower-cased host names; ``None`` means "any" (``*``)."""
    items = {h.strip().lower().strip("[]") for h in value.split(",") if h.strip()}
    if "*" in items:
        return None
    return frozenset(items)


def host_only(host_header: str) -> str:
    """Strip the port from a Host header value (IPv6-aware)."""
    value = host_header.strip().lower()
    if value.startswith("["):
        return value[1 : value.find("]")] if "]" in value else value.strip("[]")
    if value.count(":") == 1:
        return value.split(":", 1)[0]
    return value


class SameOriginGuard:
    """Pure ASGI middleware (covers both HTTP and WebSocket scopes)."""

    def __init__(self, app: ASGIApp, *, allowed_hosts: frozenset[str] | None) -> None:
        self.app = app
        self.allowed_hosts = allowed_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        host = headers.get("host", "")
        reason = self._check(scope, headers, host)
        if reason is None:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            # Refuse the handshake; Starlette maps this to a 403 response.
            await send({"type": "websocket.close", "code": 1008, "reason": reason})
            return
        body = json.dumps({"detail": reason}).encode()
        start: Message = {
            "type": "http.response.start",
            "status": 403,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
        await send(start)
        await send({"type": "http.response.body", "body": body})

    def _check(self, scope: Scope, headers: dict[str, str], host: str) -> str | None:
        if self.allowed_hosts is not None and host_only(host) not in self.allowed_hosts:
            return f"Host {host!r} is not allowed (see THINKTANK_ALLOWED_HOSTS)"
        is_ws = scope["type"] == "websocket"
        method = str(scope.get("method", "GET")).upper()
        if not is_ws and method in SAFE_METHODS:
            return None
        if headers.get("sec-fetch-site", "").lower() == "cross-site":
            return "Cross-site request refused"
        origin = headers.get("origin")
        if origin is None:
            return None  # non-browser client
        parsed = urlsplit(origin)
        if origin == "null" or not parsed.netloc or parsed.netloc.lower() != host.strip().lower():
            return "Cross-origin request refused"
        return None
