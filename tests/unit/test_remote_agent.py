"""Unit tests for :class:`RemoteAgent` (M7) using ``httpx.MockTransport``.

Covers the protocol mapping (speak/observe/session_end) and the failure
contract: a failed or empty ``speak`` raises (so the engine's error path
applies), while ``observe``/``session_end`` swallow errors.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from thinktank.domain.events import new_message_id
from thinktank.domain.models import Message, RemoteConfig
from thinktank.participants.base import TurnContext
from thinktank.participants.remote import RemoteAgent

SESSION_ID = uuid.uuid4()


def _cfg() -> RemoteConfig:
    return RemoteConfig(id="remote1", url="http://remote.test/base")


def _turn() -> TurnContext:
    return TurnContext(session_id=SESSION_ID, round=1, turn_instruction="state your view")


class _RecordingTransport:
    """Mock transport that records paths and returns canned JSON per suffix."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.speak_status = 200
        self.speak_body: dict[str, object] = {"content": "I am the remote agent; I disagree."}

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        if path.endswith("/speak"):
            if self.speak_status != 200:
                return httpx.Response(self.speak_status, json={"error": "boom"})
            return httpx.Response(200, json=self.speak_body)
        return httpx.Response(200, json={})


@pytest.mark.asyncio
async def test_speak_posts_context_and_returns_message() -> None:
    rec = _RecordingTransport()
    client = httpx.AsyncClient(transport=httpx.MockTransport(rec.handle))
    agent = RemoteAgent(_cfg(), client=client)

    msg = await agent.speak(_turn())

    assert msg is not None
    assert msg.speaker_id == "remote1"
    assert msg.kind == "speech"
    assert msg.content == "I am the remote agent; I disagree."
    assert msg.session_id == SESSION_ID
    assert isinstance(msg.id, uuid.UUID)
    assert rec.paths == ["/base/speak"]


@pytest.mark.asyncio
async def test_speak_http_error_raises() -> None:
    rec = _RecordingTransport()
    rec.speak_status = 500
    client = httpx.AsyncClient(transport=httpx.MockTransport(rec.handle))
    agent = RemoteAgent(_cfg(), client=client)

    with pytest.raises(httpx.HTTPStatusError):
        await agent.speak(_turn())


@pytest.mark.asyncio
async def test_speak_missing_content_raises() -> None:
    rec = _RecordingTransport()
    rec.speak_body = {"note": "no content field"}
    client = httpx.AsyncClient(transport=httpx.MockTransport(rec.handle))
    agent = RemoteAgent(_cfg(), client=client)

    with pytest.raises(ValueError, match="no content"):
        await agent.speak(_turn())


@pytest.mark.asyncio
async def test_observe_is_best_effort_and_never_raises() -> None:
    rec = _RecordingTransport()
    rec.speak_status = 500  # poison observe too
    client = httpx.AsyncClient(transport=httpx.MockTransport(rec.handle))
    agent = RemoteAgent(_cfg(), client=client)

    msg = Message(
        id=new_message_id(),
        session_id=SESSION_ID,
        seq=1,
        speaker_id="a",
        kind="speech",
        content="hello",
    )
    # Must not raise even though the transport returns 500.
    await agent.observe(msg)
    assert rec.paths == ["/base/observe"]


@pytest.mark.asyncio
async def test_session_end_posts_once() -> None:
    rec = _RecordingTransport()
    client = httpx.AsyncClient(transport=httpx.MockTransport(rec.handle))
    agent = RemoteAgent(_cfg(), client=client)

    # No session bound yet: nothing to report.
    await agent.on_session_end()
    assert rec.paths == []

    await agent.speak(_turn())
    await agent.on_session_end()
    assert rec.paths == ["/base/speak", "/base/session_end"]


@pytest.mark.asyncio
async def test_sends_headers_and_timeout_configured() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization", "")
        return httpx.Response(200, json={"content": "ok"})

    cfg = RemoteConfig(id="r", url="http://remote.test", display_name="R", headers={"Authorization": "Bearer tok"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    agent = RemoteAgent(cfg, client=client)

    msg = await agent.speak(_turn())
    assert msg is not None
    assert agent.display_name == "R"
    assert seen["auth"] == "Bearer tok"
