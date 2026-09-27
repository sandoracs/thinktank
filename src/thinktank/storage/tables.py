"""SQLAlchemy 2.0 ORM models.

These are the *persisted* projections and records. The ``events`` table is the
source of truth; ``messages``, ``persona_versions`` etc. are rebuilt from it.
UUIDs are stored as CHAR(36) for portability across aiosqlite versions.

The vector (``memory_vec``) and full-text (``memory_fts`` / ``messages_fts``)
tables are NOT ORM models — they are SQLite virtual tables created via raw SQL
in :mod:`thinktank.storage.db`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class AgentTemplate(Base):
    __tablename__ = "agent_templates"

    id: Mapped[str] = mapped_column(sa.String(128), primary_key=True)
    config: Mapped[dict[str, Any]] = mapped_column(sa.JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class SessionTemplate(Base):
    __tablename__ = "session_templates"

    id: Mapped[str] = mapped_column(sa.String(128), primary_key=True)
    config: Mapped[dict[str, Any]] = mapped_column(sa.JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True)
    title: Mapped[str] = mapped_column(sa.Text)
    status: Mapped[str] = mapped_column(sa.String(32), default="created")
    config: Mapped[dict[str, Any]] = mapped_column(sa.JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("sessions.id"), nullable=False)
    seq: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    type: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(sa.JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        sa.UniqueConstraint("session_id", "seq", name="uq_events_session_seq"),
        sa.Index("ix_events_session_type", "session_id", "type"),
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("sessions.id"), nullable=False)
    seq: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    speaker_id: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    kind: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    reply_to: Mapped[str | None] = mapped_column(sa.String(36), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(sa.JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=utcnow)


class PersonaVersion(Base):
    __tablename__ = "persona_versions"

    agent_id: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    session_id: Mapped[str] = mapped_column(sa.String(36), nullable=False)
    version: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    state: Mapped[dict[str, Any]] = mapped_column(sa.JSON, default=dict)
    cause_seq: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=utcnow)

    __table_args__ = (sa.PrimaryKeyConstraint("agent_id", "session_id", "version"),)


class MemoryItem(Base):
    __tablename__ = "memory_items"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(sa.String(128), nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(sa.String(36), nullable=True)
    layer: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    source_seq: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(sa.JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=utcnow)


class PendingApproval(Base):
    __tablename__ = "pending_approvals"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(sa.String(36), nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    proposal: Mapped[dict[str, Any]] = mapped_column(sa.JSON, default=dict)
    status: Mapped[str] = mapped_column(sa.String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=utcnow)
    decided_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)


class AppMeta(Base):
    __tablename__ = "app_meta"

    key: Mapped[str] = mapped_column(sa.String(128), primary_key=True)
    value: Mapped[str] = mapped_column(sa.Text, nullable=False)
