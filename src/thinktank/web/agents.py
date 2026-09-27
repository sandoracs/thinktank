"""Agent-template persistence (DESIGN.md §7 ``agent_templates``, §15 library).

The library is the DB-backed source of agent definitions for the web hub. On
first start the built-in defaults are seeded; ``POST /api/sessions`` resolves
participant ids against this library.
"""

from __future__ import annotations

import colorsys
import random

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from thinktank.domain.models import AgentConfig
from thinktank.storage.tables import AgentTemplate
from thinktank.web.defaults import default_agents


def random_color() -> str:
    """A random but pleasant hex color for one agent's display.

    Random hue, fixed medium lightness/saturation, so the result is always
    distinct enough and the transcript can pick a readable text color for it.
    """
    r, g, b = colorsys.hls_to_rgb(random.random(), 0.55, 0.65)  # noqa: S311 (display color, not crypto)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"

async def count_templates(factory: async_sessionmaker[AsyncSession]) -> int:
    async with factory() as db:
        return (await db.execute(select(func.count()).select_from(AgentTemplate))).scalar_one()


async def seed_defaults(factory: async_sessionmaker[AsyncSession]) -> int:
    """Insert the built-in agents if the library is empty. Returns rows added."""
    if await count_templates(factory) > 0:
        return 0
    async with factory() as db, db.begin():
        for config in default_agents().values():
            db.add(AgentTemplate(id=config.id, config=config.model_dump(mode="json")))
    return len(default_agents())


async def ensure_colors(factory: async_sessionmaker[AsyncSession]) -> int:
    """Give every template that has no display color a random one. Returns rows updated."""
    async with factory() as db, db.begin():
        rows = (await db.execute(select(AgentTemplate))).scalars().all()
        changed = 0
        for row in rows:
            cfg = AgentConfig.model_validate(row.config)
            if cfg.color is None:
                row.config = cfg.model_copy(update={"color": random_color()}).model_dump(mode="json")
                changed += 1
        return changed


async def list_templates(factory: async_sessionmaker[AsyncSession]) -> list[AgentConfig]:
    async with factory() as db:
        rows = (await db.execute(select(AgentTemplate).order_by(AgentTemplate.id))).scalars().all()
    return [AgentConfig.model_validate(row.config) for row in rows]


async def get_template(
    factory: async_sessionmaker[AsyncSession], agent_id: str
) -> AgentConfig | None:
    async with factory() as db:
        row = await db.get(AgentTemplate, agent_id)
    return AgentConfig.model_validate(row.config) if row else None


async def resolve_agents(
    factory: async_sessionmaker[AsyncSession], agent_ids: list[str]
) -> dict[str, AgentConfig]:
    """Load every referenced template; raise if any is missing (fail fast)."""
    agents: dict[str, AgentConfig] = {}
    missing = []
    for agent_id in agent_ids:
        config = await get_template(factory, agent_id)
        if config is None:
            missing.append(agent_id)
        else:
            agents[agent_id] = config
    if missing:
        msg = f"Unknown agent template(s): {', '.join(missing)}"
        raise ValueError(msg)
    return agents


async def create_template(
    factory: async_sessionmaker[AsyncSession], config: AgentConfig
) -> AgentConfig:
    """Insert a new agent template; raise if the id already exists.

    A random display color is assigned when the config does not carry one.
    """
    if config.color is None:
        config = config.model_copy(update={"color": random_color()})
    async with factory() as db, db.begin():
        if await db.get(AgentTemplate, config.id) is not None:
            msg = f"Agent template {config.id!r} already exists"
            raise ValueError(msg)
        db.add(AgentTemplate(id=config.id, config=config.model_dump(mode="json")))
    return config


async def update_template(
    factory: async_sessionmaker[AsyncSession], config: AgentConfig
) -> AgentConfig | None:
    """Replace an existing template's config; return None if it does not exist."""
    async with factory() as db, db.begin():
        row = await db.get(AgentTemplate, config.id)
        if row is None:
            return None
        if config.color is None:
            existing = AgentConfig.model_validate(row.config)
            if existing.color is not None:
                config = config.model_copy(update={"color": existing.color})
        row.config = config.model_dump(mode="json")
    return config


async def delete_template(
    factory: async_sessionmaker[AsyncSession], agent_id: str
) -> bool:
    """Delete a template; return whether a row was removed."""
    async with factory() as db, db.begin():
        row = await db.get(AgentTemplate, agent_id)
        if row is None:
            return False
        await db.delete(row)
    return True


def template_summary(config: AgentConfig) -> dict[str, object]:
    """JSON-safe view for the persona templates (DESIGN.md §15)."""
    persona = config.persona
    return {
        "id": config.id,
        "model": config.model,
        "temperature": config.temperature,
        "persona": {
            "name": persona.name,
            "role": persona.role,
            "age": persona.age,
            "expertise": persona.expertise,
            "values": persona.values,
            "boundaries": persona.boundaries,
        },
        "drift_mode": config.drift.mode.value,
        "color": config.color,
        "consistency_check": config.consistency_check,
        "consistency_threshold": config.consistency_threshold,
        "memory": {
            "working_window": config.memory.working_window,
            "summarize_every": config.memory.summarize_every,
            "retrieval_k": config.memory.retrieval_k,
            "long_term": config.memory.long_term,
        },
        "initial_state": {
            "mood": config.initial_state.mood,
            "stances": {k: v.model_dump(mode="json") for k, v in config.initial_state.stances.items()},
            "attitudes": dict(config.initial_state.attitudes),
        },
    }


