"""Repositories over the async session factory (DESIGN.md §3, §7).

The :class:`EventStore` is the single writer for the event stream: it assigns
the per-session ``seq``, persists the row, and maintains the ``messages``
projection — all inside one transaction. Sequence assignment is made safe by
the :class:`EventBus` lock, which serialises appends per bus, so the
read-max-then-insert here never races within a session.
"""

from __future__ import annotations

import uuid

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from roundtable.core.state import (
    STATUS_ENDED,
    STATUS_INTERRUPTED,
    STATUS_PAUSED,
    STATUS_RUNNING,
)
from roundtable.domain.events import Event, EventType, MessagePostedPayload
from roundtable.domain.models import Message, SessionConfig
from roundtable.storage.tables import Event as EventRow
from roundtable.storage.tables import Message as MessageRow
from roundtable.storage.tables import Session as SessionRow

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


def event_from_row(row: EventRow) -> Event:
    return Event(seq=row.seq, type=EventType(row.type), payload=row.payload, created_at=row.created_at)
