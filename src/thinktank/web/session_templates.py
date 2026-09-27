"""Session-template persistence (DESIGN.md §7 ``session_templates``, §15).

A saved session is a reusable configuration: the builder's "save as template"
stores the current form as a template, and the builder can prefill from one.
The stored config is a full :class:`SessionConfig` snapshot, so a template
reproduces the session exactly as it was configured.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from thinktank.domain.models import SessionConfig
from thinktank.storage.tables import SessionTemplate


async def create_template(
    factory: async_sessionmaker[AsyncSession], template_id: str, config: SessionConfig
) -> SessionConfig:
    """Store a new session template; raise if the id already exists."""
    async with factory() as db, db.begin():
        if await db.get(SessionTemplate, template_id) is not None:
            msg = f"Session template {template_id!r} already exists"
            raise ValueError(msg)
        db.add(SessionTemplate(id=template_id, config=config.model_dump(mode="json")))
    return config


async def update_template(
    factory: async_sessionmaker[AsyncSession], template_id: str, config: SessionConfig
) -> SessionConfig | None:
    """Replace an existing template's config; return None if it does not exist."""
    async with factory() as db, db.begin():
        row = await db.get(SessionTemplate, template_id)
        if row is None:
            return None
        row.config = config.model_dump(mode="json")
    return config


async def list_templates(
    factory: async_sessionmaker[AsyncSession],
) -> list[tuple[str, SessionConfig]]:
    async with factory() as db:
        rows = (await db.execute(select(SessionTemplate).order_by(SessionTemplate.id))).scalars().all()
    return [(row.id, SessionConfig.model_validate(row.config)) for row in rows]


async def get_template(
    factory: async_sessionmaker[AsyncSession], template_id: str
) -> SessionConfig | None:
    async with factory() as db:
        row = await db.get(SessionTemplate, template_id)
    return SessionConfig.model_validate(row.config) if row else None


async def delete_template(factory: async_sessionmaker[AsyncSession], template_id: str) -> bool:
    async with factory() as db, db.begin():
        row = await db.get(SessionTemplate, template_id)
        if row is None:
            return False
        await db.delete(row)
    return True
