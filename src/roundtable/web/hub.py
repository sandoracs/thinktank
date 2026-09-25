"""Web hub: in-process registry of live sessions for the FastAPI layer
(DESIGN.md §14, §15).

Owns per-session live state that outlives a single request: the running
:class:`SessionEngine`, the ``HumanParticipant`` instances the WebSocket
layer drives, per-human access tokens, and the set of open WebSocket
connections. The engine and event store do the heavy lifting; the hub is the
thin coordination layer between HTTP/WebSocket handlers and them.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from roundtable.core.engine import SessionEngine
from roundtable.core.manager import SessionManager
from roundtable.core.state import (
    STATUS_ENDED,
    STATUS_PAUSED,
    STATUS_RUNNING,
)
from roundtable.domain.events import EventType
from roundtable.domain.models import AgentConfig, SessionConfig
from roundtable.participants.human import HumanParticipant


@dataclass
class HubSession:
    """One live (or recently live) session as seen by the hub."""

    session_id: uuid.UUID
    engine: SessionEngine
    humans: dict[str, HumanParticipant] = field(default_factory=dict)
    tokens: dict[str, str] = field(default_factory=dict)  # participant_id -> token
    task: asyncio.Task[None] | None = None


class WebHub:
    def __init__(self, manager: SessionManager) -> None:
        self._manager = manager
        self._sessions: dict[uuid.UUID, HubSession] = {}
        self._ws: dict[uuid.UUID, set[WebSocket]] = {}

    # -- session lifecycle ---------------------------------------------------
    async def create_session(self, config: SessionConfig, agents: dict[str, AgentConfig]) -> HubSession:
        """Create the session (status ``created``) without starting the engine."""
        engine = await self._manager.create_session(config, agents)
        humans = {
            pid: p for pid, p in engine.participants.items() if isinstance(p, HumanParticipant)
        }
        entry = HubSession(session_id=engine.session_id, engine=engine, humans=humans)
        for pid in humans:
            entry.tokens[pid] = secrets.token_urlsafe(16)
        self._sessions[entry.session_id] = entry
        return entry

    def start(self, session_id: uuid.UUID) -> bool:
        """Begin the engine loop if this session has not started yet."""
        entry = self._sessions.get(session_id)
        if entry is None or entry.task is not None or entry.engine.state.status != "created":
            return False
        entry.task = asyncio.create_task(self._manager.start(entry.engine), name=f"session-{session_id}")
        entry.task.add_done_callback(lambda _t: self._forget(session_id))
        return True


    def session(self, session_id: uuid.UUID) -> HubSession | None:
        return self._sessions.get(session_id)

    def engine(self, session_id: uuid.UUID) -> SessionEngine | None:
        entry = self._sessions.get(session_id)
        return entry.engine if entry else None

    def human(self, session_id: uuid.UUID, participant_id: str) -> HumanParticipant | None:
        entry = self._sessions.get(session_id)
        return entry.humans.get(participant_id) if entry else None

    # -- tokens --------------------------------------------------------------
    def issue_token(self, session_id: uuid.UUID, participant_id: str) -> str | None:
        entry = self._sessions.get(session_id)
        if entry is None or participant_id not in entry.humans:
            return None
        return entry.tokens.setdefault(participant_id, secrets.token_urlsafe(16))

    def resolve_token(self, session_id: uuid.UUID, token: str) -> str | None:
        """Map a token back to its participant id (or ``None`` if invalid)."""
        entry = self._sessions.get(session_id)
        if entry is None:
            return None
        for pid, value in entry.tokens.items():
            if secrets.compare_digest(value, token):
                return pid
        return None

    # -- WebSocket registry ----------------------------------------------------
    def add_ws(self, session_id: uuid.UUID, ws: WebSocket) -> None:
        self._ws.setdefault(session_id, set()).add(ws)

    def drop_ws(self, session_id: uuid.UUID, ws: WebSocket) -> None:
        connections = self._ws.get(session_id)
        if connections:
            connections.discard(ws)

    async def broadcast(self, session_id: uuid.UUID, message: dict[str, Any]) -> None:
        for ws in list(self._ws.get(session_id, ())):
            await self._safe_send(ws, message)

    async def send_to(self, session_id: uuid.UUID, participant_id: str, message: dict[str, Any]) -> None:
        for ws in list(self._ws.get(session_id, ())):
            if getattr(ws, "roundtable_participant", None) == participant_id:
                await self._safe_send(ws, message)

    @staticmethod
    async def _safe_send(ws: WebSocket, message: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):  # disconnects are normal; drop silently
            await ws.send_json(message)

    # -- controls (pause / resume / stop) --------------------------------------
    async def pause(self, session_id: uuid.UUID) -> bool:
        engine = self.engine(session_id)
        if engine is None or engine.state.status != STATUS_RUNNING:
            return False
        self._cancel_human_turns(session_id)
        await engine.emit(EventType.SESSION_PAUSED)
        engine.pause()
        return True

    async def resume(self, session_id: uuid.UUID) -> bool:
        engine = self.engine(session_id)
        if engine is None or engine.state.status != STATUS_PAUSED:
            return False
        await engine.emit(EventType.SESSION_RESUMED)  # persists + flips DB status
        engine.resume()
        return True

    def stop(self, session_id: uuid.UUID) -> bool:
        engine = self.engine(session_id)
        if engine is None or engine.state.status in (STATUS_ENDED, "interrupted"):
            return False
        self._cancel_human_turns(session_id)
        engine.stop("manual")
        return True

    def _cancel_human_turns(self, session_id: uuid.UUID) -> None:
        """Abandon any in-flight human turn so pause/stop takes effect promptly."""
        entry = self._sessions.get(session_id)
        if entry is None:
            return
        for human in entry.humans.values():
            human.cancel_turn()

    # -- internal --------------------------------------------------------------
    def _forget(self, session_id: uuid.UUID) -> None:
        self._sessions.pop(session_id, None)
        self._ws.pop(session_id, None)

    async def shutdown(self) -> None:
        """Cancel and await any still-running session tasks (call at app teardown).

        A live engine task may be holding an open DB connection; letting the app
        dispose the engine while the task is still alive leaks the aiosqlite
        connection (it is garbage-collected before ``close()`` runs).
        """
        tasks = [entry.task for entry in self._sessions.values() if entry.task is not None]
        for entry in self._sessions.values():
            if entry.task is not None:
                entry.task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._sessions.clear()
        self._ws.clear()
