"""Repositories over the async session factory.

The :class:`EventStore` is the single writer for the event stream: it assigns
the per-session ``seq``, persists the row, and maintains the ``messages``
projection — all inside one transaction. Sequence assignment is made safe by
the :class:`EventBus` lock, which serialises appends per bus, so the
read-max-then-insert here never races within a session.
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from thinktank.core.state import (
    STATUS_CREATED,
    STATUS_ENDED,
    STATUS_INTERRUPTED,
    STATUS_PAUSED,
    STATUS_RUNNING,
)
from thinktank.domain.events import Event, EventType, MessagePostedPayload
from thinktank.domain.models import Message, SessionConfig
from thinktank.storage.tables import Event as EventRow
from thinktank.storage.tables import Message as MessageRow
from thinktank.storage.tables import PersonaVersion as PersonaVersionRow
from thinktank.storage.tables import Session as SessionRow

_LIFECYCLE = {
    EventType.SESSION_STARTED: STATUS_RUNNING,
    EventType.SESSION_PAUSED: STATUS_PAUSED,
    EventType.SESSION_RESUMED: STATUS_RUNNING,
    EventType.SESSION_ENDED: STATUS_ENDED,
}


class EventStore:
    """Append-only event store plus the message projection and session records."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory: async_sessionmaker[AsyncSession] = factory

    # -- session lifecycle -------------------------------------------------
    async def create_session(self, session_id: uuid.UUID, config: SessionConfig) -> None:
        async with self._factory() as db, db.begin():
            db.add(
                SessionRow(
                    id=str(session_id),
                    title=config.title,
                    status="created",
                    config=config.model_dump(mode="json"),
                )
            )

    async def set_status(self, session_id: uuid.UUID, status: str) -> None:
        async with self._factory() as db, db.begin():
            await db.execute(
                sa.update(SessionRow).where(SessionRow.id == str(session_id)).values(status=status)
            )

    async def update_session_config(self, session_id: uuid.UUID, config: SessionConfig) -> None:
        """Replace the stored config (only meaningful while the session is ``created``)."""
        async with self._factory() as db, db.begin():
            await db.execute(
                sa.update(SessionRow)
                .where(SessionRow.id == str(session_id))
                .values(config=config.model_dump(mode="json"))
            )

    async def get_session(self, session_id: uuid.UUID) -> SessionRow | None:
        async with self._factory() as db:
            return await db.get(SessionRow, str(session_id))

    async def list_sessions(self) -> list[SessionRow]:
        async with self._factory() as db:
            result = await db.execute(sa.select(SessionRow).order_by(SessionRow.created_at.desc()))
            return list(result.scalars().all())

    async def mark_interrupted_on_start(self) -> int:
        """Crash recovery: any session left ``running`` becomes ``interrupted``."""
        async with self._factory() as db, db.begin():
            result = await db.execute(
                sa.update(SessionRow).where(SessionRow.status == STATUS_RUNNING).values(status=STATUS_INTERRUPTED)
            )
            return result.rowcount or 0  # pyright: ignore[reportUnknownVariableType,reportAttributeAccessIssue]

    async def reset_session(self, session_id: uuid.UUID) -> None:
        """Wipe the stored conversation (events + projections) and restore initial state.

        Memory items are intentionally kept: they are the agent's accumulated
        research knowledge, not part of the conversation being reset.
        """
        session_key = str(session_id)
        from thinktank.storage.tables import PendingApproval as PendingApprovalRow

        async with self._factory() as db, db.begin():
            await db.execute(sa.delete(EventRow).where(EventRow.session_id == session_key))
            await db.execute(sa.delete(MessageRow).where(MessageRow.session_id == session_key))
            await db.execute(sa.delete(PersonaVersionRow).where(PersonaVersionRow.session_id == session_key))
            await db.execute(sa.delete(PendingApprovalRow).where(PendingApprovalRow.session_id == session_key))
            await db.execute(
                sa.update(SessionRow)
                .where(SessionRow.id == session_key)
                .values(status=STATUS_CREATED, ended_at=None)
            )

    async def delete_session(self, session_id: uuid.UUID) -> bool:
        """Delete the session row and all its stored conversation data.

        Memory items are intentionally kept (same as :meth:`reset_session`):
        they are the agent's accumulated knowledge, not part of the session.
        Returns ``False`` when the session does not exist.
        """
        session_key = str(session_id)
        from thinktank.storage.tables import PendingApproval as PendingApprovalRow

        async with self._factory() as db, db.begin():
            row = await db.get(SessionRow, session_key)
            if row is None:
                return False
            await db.execute(sa.delete(EventRow).where(EventRow.session_id == session_key))
            await db.execute(sa.delete(MessageRow).where(MessageRow.session_id == session_key))
            await db.execute(sa.delete(PersonaVersionRow).where(PersonaVersionRow.session_id == session_key))
            await db.execute(sa.delete(PendingApprovalRow).where(PendingApprovalRow.session_id == session_key))
            await db.delete(row)
            return True

    # -- events ------------------------------------------------------------
    async def append(self, session_id: uuid.UUID, event: Event) -> Event:
        """Assign ``seq``, persist the event (and message projection), atomically."""
        session_key = str(session_id)
        async with self._factory() as db, db.begin():
            max_row = await db.execute(
                sa.select(sa.func.max(EventRow.seq)).where(EventRow.session_id == session_key)
            )
            current_max = max_row.scalar() or 0
            seq = current_max + 1
            stored = EventRow(
                session_id=session_key,
                seq=seq,
                type=event.type.value,
                payload=event.payload,
                created_at=event.created_at,
            )
            db.add(stored)

            if event.type is EventType.MESSAGE_POSTED:
                message: Message = event.payload_as(MessagePostedPayload).message
                db.add(
                    MessageRow(
                        id=str(message.id),
                        session_id=session_key,
                        seq=seq,
                        speaker_id=message.speaker_id,
                        kind=message.kind,
                        content=message.content,
                        reply_to=str(message.reply_to) if message.reply_to else None,
                        meta=message.meta,
                        created_at=message.created_at or event.created_at,
                    )
                )

            if event.type is EventType.APPROVAL_REQUESTED:
                from thinktank.storage.tables import PendingApproval as PendingApprovalRow

                db.add(
                    PendingApprovalRow(
                        session_id=session_key,
                        agent_id=event.payload.get("agent_id", ""),
                        proposal=event.payload.get("proposal", {}),
                        status="pending",
                    )
                )

            if event.type in _LIFECYCLE:
                await db.execute(
                    sa.update(SessionRow).where(SessionRow.id == session_key).values(status=_LIFECYCLE[event.type])
                )

        return Event(seq=seq, type=event.type, payload=event.payload, created_at=event.created_at)

    async def get_events(
        self,
        session_id: uuid.UUID,
        after_seq: int = 0,
        types: set[EventType] | None = None,
    ) -> list[Event]:
        session_key = str(session_id)
        async with self._factory() as db:
            query = (
                sa.select(EventRow)
                .where(EventRow.session_id == session_key, EventRow.seq > after_seq)
                .order_by(EventRow.seq)
            )
            if types:
                query = query.where(EventRow.type.in_([t.value for t in types]))
            rows = (await db.execute(query)).scalars().all()
        return [
            Event(seq=r.seq, type=EventType(r.type), payload=r.payload, created_at=r.created_at)
            for r in rows
        ]

    async def last_seq(self, session_id: uuid.UUID) -> int:
        async with self._factory() as db:
            row = await db.execute(
                sa.select(sa.func.max(EventRow.seq)).where(EventRow.session_id == str(session_id))
            )
            return row.scalar() or 0

    # -- persona versions --------------------------------------------------
    async def next_persona_version(self, agent_id: str, session_id: uuid.UUID) -> int:
        async with self._factory() as db:
            row = await db.execute(
                sa.select(sa.func.coalesce(sa.func.max(PersonaVersionRow.version), 0)).where(
                    PersonaVersionRow.agent_id == agent_id,
                    PersonaVersionRow.session_id == str(session_id),
                )
            )
            return int(row.scalar() or 0) + 1

    async def append_persona_version(
        self,
        agent_id: str,
        session_id: uuid.UUID,
        version: int,
        state: dict[str, object],
        cause_seq: int | None,
    ) -> None:
        async with self._factory() as db, db.begin():
            db.add(
                PersonaVersionRow(
                    agent_id=agent_id,
                    session_id=str(session_id),
                    version=version,
                    state=state,
                    cause_seq=cause_seq,
                )
            )

    async def persona_history(
        self, agent_id: str, session_id: uuid.UUID
    ) -> list[PersonaVersionRow]:
        async with self._factory() as db:
            rows = (
                await db.execute(
                    sa.select(PersonaVersionRow)
                    .where(
                        PersonaVersionRow.agent_id == agent_id,
                        PersonaVersionRow.session_id == str(session_id),
                    )
                    .order_by(PersonaVersionRow.version)
                )
            ).scalars().all()
        return list(rows)

    # -- approvals (M6) ---------------------------
    async def list_approvals(
        self, session_id: uuid.UUID, agent_id: str | None = None
    ) -> list[dict[str, object]]:
        """Return the session's approval requests (pending first), newest last."""
        from thinktank.storage.tables import PendingApproval as PendingApprovalRow

        async with self._factory() as db:
            stmt = (
                sa.select(PendingApprovalRow)
                .where(PendingApprovalRow.session_id == str(session_id))
                .order_by(PendingApprovalRow.created_at, PendingApprovalRow.id)
            )
            if agent_id is not None:
                stmt = stmt.where(PendingApprovalRow.agent_id == agent_id)
            rows = (await db.execute(stmt)).scalars().all()
        return [
            {
                "id": r.id,
                "agent_id": r.agent_id,
                "proposal": r.proposal,
                "status": r.status,
                "created_at": r.created_at,
                "decided_at": r.decided_at,
            }
            for r in rows
        ]

    async def decide_approval_row(self, row_id: int, status: str) -> bool:
        """Mark exactly the pending approval identified by ``row_id`` as decided.

        Scoped to a single row (not just ``pending``) so deciding one proposal
        never touches another pending approval for the same agent.
        """
        from thinktank.storage.tables import PendingApproval as PendingApprovalRow

        async with self._factory() as db, db.begin():
            result = await db.execute(
                sa.update(PendingApprovalRow)
                .where(
                    PendingApprovalRow.id == row_id,
                    PendingApprovalRow.status == "pending",
                )
                .values(status=status, decided_at=datetime.now(UTC))
            )
            return (result.rowcount or 0) > 0  # pyright: ignore[reportUnknownVariableType,reportAttributeAccessIssue]

    # -- export (M6) -------------------------------------
    async def export(self, session_id: uuid.UUID, format: str = "jsonl") -> str:
        """Render the full event stream as ``jsonl`` or ``csv`` text."""
        events = await self.get_events(session_id)
        return render_export(events, str(session_id), format)


def render_export(events: list[Event], session_id: str, format: str) -> str:
    """Render events as analysis-ready ``jsonl`` or ``csv``.

    JSONL: one JSON object per event, with message payloads inlined so the
    transcript is directly usable. CSV: a flat transcript of the spoken turns.
    """
    if format not in ("jsonl", "csv"):
        raise ValueError(f"unsupported export format: {format!r}")
    ordered = sorted(events, key=lambda e: e.seq)
    if format == "jsonl":
        lines = [
            json.dumps(_event_record(e, session_id), ensure_ascii=False, default=str)
            for e in ordered
        ]
        return "\n".join(lines) + ("\n" if lines else "")

    # CSV transcript
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["seq", "created_at", "type", "speaker", "content"])
    for e in ordered:
        if e.type is EventType.MESSAGE_POSTED:
            message = e.payload.get("message") or {}
            writer.writerow(
                [e.seq, e.created_at, e.type, message.get("speaker_id", ""), message.get("content", "")]
            )
        else:
            writer.writerow([e.seq, e.created_at, e.type, "", ""])
    return buf.getvalue()


def _event_record(e: Event, session_id: str) -> dict[str, object]:
    """Flatten an event into an analysis-ready record."""
    record: dict[str, object] = {
        "session_id": session_id,
        "seq": e.seq,
        "type": e.type,
        "created_at": e.created_at,
    }
    if e.type is EventType.MESSAGE_POSTED:
        message = e.payload.get("message") or {}
        record["speaker"] = message.get("speaker_id")
        record["kind"] = message.get("kind")
        record["content"] = message.get("content")
    elif e.type is EventType.TURN_ASSIGNED:
        record["speaker"] = e.payload.get("speaker_id")
        record["strategy"] = e.payload.get("strategy")
    elif e.type in (
        EventType.PERSONA_UPDATED,
        EventType.PERSONA_UPDATE_CLAMPED,
        EventType.PERSONA_UPDATE_REJECTED,
        EventType.APPROVAL_REQUESTED,
        EventType.APPROVAL_DECIDED,
    ):
        record["agent"] = e.payload.get("agent_id")
        if e.payload.get("proposal") is not None:
            record["proposal"] = e.payload["proposal"]
        if e.payload.get("state") is not None:
            record["state"] = e.payload["state"]
        if e.payload.get("reason") is not None:
            record["reason"] = e.payload["reason"]
        if e.payload.get("notes") is not None:
            record["notes"] = e.payload["notes"]
    elif e.type is EventType.LLM_CALL_COMPLETED:
        record["model"] = e.payload.get("model")
        record["cost_usd"] = e.payload.get("cost_usd")
        record["input_tokens"] = e.payload.get("input_tokens")
        record["output_tokens"] = e.payload.get("output_tokens")
    else:
        record["payload"] = e.payload
    return record


def event_from_row(row: EventRow) -> Event:
    return Event(seq=row.seq, type=EventType(row.type), payload=row.payload, created_at=row.created_at)
