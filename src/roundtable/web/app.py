"""FastAPI hub application (DESIGN.md §14, §15).

Wires the M1 core (manager / store / engine) to HTTP: REST for control and
data, one WebSocket per session for the lossless live feed (replay from
``after_seq`` then live), and Jinja2+HTMX pages for the dashboard, agent
library, session builder, and the live table.

The app is built by :func:`create_app` so tests (and the CLI) can inject a
:class:`~roundtable.llm.client.LLMClient` (e.g. ``FakeLLM``) and a throwaway
database.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.datastructures import FormData

from roundtable.config import Settings, get_settings
from roundtable.core.manager import SessionManager
from roundtable.core.state import SessionState, apply_event
from roundtable.domain.events import Event, EventType
from roundtable.domain.models import (
    AgentConfig,
    DriftMode,
    Layer,
    PersonaCore,
    SessionConfig,
)
from roundtable.llm.client import LLMClient
from roundtable.memory.base import MemoryBackend
from roundtable.memory.embeddings import EmbeddingProvider, build_embedding_provider
from roundtable.memory.sqlite import SQLiteMemoryBackend
from roundtable.plugins.registry import load_strategies
from roundtable.storage.db import (
    dispose,
    init_db,
    init_memory_tables,
    make_engine,
    make_session_factory,
    stored_embedding_model,
)
from roundtable.storage.repositories import EventStore
from roundtable.web import agents as agent_repo
from roundtable.web import session_templates as session_tpl
from roundtable.web.hub import HubSession, WebHub
from roundtable.web.render import sidebar_fragment, transcript_fragment

_WEB = Path(__file__).parent
_TEMPLATES = _WEB / "templates"

logger = logging.getLogger(__name__)

def _deps(request: Request | WebSocket) -> dict[str, Any]:
    """Shared dependencies stored on ``app.state``."""
    return request.app.state.deps


def _parse_session_id(raw: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(raw)
    except ValueError:
        return None


def _page(template: str, context: dict[str, Any]) -> HTMLResponse:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(("html",)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    html = env.get_template(template).render(**context, title=context.get("title", "Roundtable"))
    return HTMLResponse(html)


def embedding_mismatch_message(stored: str | None, configured: str) -> str | None:
    """Warn when stored memory vectors predate the configured embedding model.

    ``None`` means no mismatch (nothing stored yet, or the model already matches).
    """
    if stored is None or stored == configured:
        return None
    return (
        f"embedding model mismatch: stored memory vectors use {stored!r} but the configured "
        f"provider is {configured!r}. Retrieval quality may be degraded; run "
        f"`roundtable reembed` to recompute the vectors."
    )


def _default_llm(settings: Settings) -> LLMClient:
    from roundtable.llm.litellm_client import LiteLLMClient

    return LiteLLMClient(timeout_s=settings.llm_timeout_s)


async def _create_session(deps: dict[str, Any], config: SessionConfig) -> HubSession:
    """Validate, resolve agent templates, and register the session in the hub."""
    agent_ids = config.agent_ids()
    try:
        agents = await agent_repo.resolve_agents(deps["factory"], agent_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await deps["hub"].create_session(config, agents)


def _multi(form: FormData, key: str) -> list[str]:
    """Normalize a possibly multi-valued form field to a list of trimmed strings."""
    values = form.getlist(key)
    return [value.strip() for value in values if isinstance(value, str) and value.strip()]

def _config_from_form(form: FormData) -> SessionConfig:
    """Build a SessionConfig from the builder form (DESIGN.md §15)."""
    title = str(form.get("title") or "").strip()
    topic = str(form.get("topic") or "").strip()
    if not title or not topic:
        raise HTTPException(status_code=400, detail="Title and topic are required")

    participants: list[dict[str, str]] = []
    for agent_id in _multi(form, "agents"):
        if agent_id:
            participants.append({"agent": agent_id})
    for human_raw in str(form.get("humans") or "").split(","):
        name = human_raw.strip()
        if name:
            participants.append({"human": name})
    if not participants:
        raise HTTPException(status_code=400, detail="At least one participant is required")

    moderator = str(form.get("moderator") or "").strip() or None
    if moderator:
        participants = [{"agent": moderator}] + [p for p in participants if p != {"agent": moderator}]

    strategy_name = str(form.get("strategy") or "round_robin")
    strategy_params: dict[str, Any] = {}
    raw_params = str(form.get("strategy_params") or "").strip()
    if raw_params:
        try:
            parsed: object = json.loads(raw_params)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="strategy_params must be JSON") from exc
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="strategy_params must be a JSON object")
        strategy_params = parsed
    try:
        max_rounds = int(str(form.get("max_rounds") or "6").strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="max_rounds must be an integer") from exc
    try:
        max_cost = float(str(form.get("max_cost_usd") or "2.0").strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="max_cost_usd must be a number") from exc

    questions: list[dict[str, str]] = []
    for i, line in enumerate(str(form.get("questions") or "").splitlines()):
        text = line.strip()
        if text:
            questions.append({"id": f"q{i + 1}", "text": text})

    raw: dict[str, Any] = {
        "title": title,
        "topic": topic,
        "language": str(form.get("language") or "hu"),
        "questions": questions,
        "participants": participants,
        "moderator": moderator,
        "strategy": {"name": strategy_name, "params": strategy_params},
        "stop": {"max_rounds": max_rounds, "max_cost_usd": max_cost},
    }
    try:
        return SessionConfig.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _slugify(text: str) -> str:
    """Derive a safe template id from a free-form title."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "session"


def _agent_from_form(form: FormData) -> AgentConfig:
    """Build an AgentConfig from the library form (DESIGN.md §15)."""

    def _csv(key: str) -> list[str]:
        return [part.strip() for part in str(form.get(key) or "").split(",") if part.strip()]

    agent_id = str(form.get("id") or "").strip()
    if not agent_id:
        raise HTTPException(status_code=400, detail="id is required")
    drift_mode = str(form.get("drift_mode") or "locked").strip()
    try:
        mode = DriftMode(drift_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"unknown drift mode {drift_mode!r}") from exc
    try:
        temperature = float(str(form.get("temperature") or "0.8").strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="temperature must be a number") from exc
    try:
        threshold = int(str(form.get("consistency_threshold") or "3").strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="consistency_threshold must be an integer") from exc

    def _int(key: str, default: int) -> int:
        raw_val = str(form.get(key) or "").strip()
        if not raw_val:
            return default
        try:
            return int(raw_val)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{key} must be an integer") from exc

    persona = PersonaCore(
        name=str(form.get("persona_name") or agent_id).strip() or agent_id,
        role=str(form.get("persona_role") or "").strip(),
        expertise=_csv("expertise"),
        values=_csv("values"),
        boundaries=_csv("boundaries"),
    )
    initial_mood = str(form.get("initial_mood") or "neutral").strip() or "neutral"
    raw: dict[str, Any] = {
        "id": agent_id,
        "model": str(form.get("model") or "").strip() or "claude-3-5-sonnet-latest",
        "temperature": temperature,
        "persona": persona,
        "drift": {"mode": mode},
        "consistency_check": (form.get("consistency_check") or "") == "on",
        "consistency_threshold": threshold,
        "memory": {
            "working_window": _int("working_window", 12),
            "summarize_every": _int("summarize_every", 8),
            "retrieval_k": _int("retrieval_k", 5),
            "long_term": (form.get("long_term") or "on") == "on",
        },
        "initial_state": {"mood": initial_mood},
    }
    try:
        return AgentConfig.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _inspector_data(
    deps: dict[str, Any], sid: uuid.UUID, agent_id: str
) -> dict[str, Any]:
    """Assemble the agent-inspector payload (DESIGN.md §15, M4)."""
    store: EventStore = deps["store"]
    manager: SessionManager = deps["manager"]
    factory = deps["factory"]
    agent = await agent_repo.get_template(factory, agent_id)
    state = await manager.get_state(sid)
    current = state.persona_states.get(agent_id)
    if current is None and agent is not None:
        current = agent.initial_state
    history = await store.persona_history(agent_id, sid)
    persona_types = {
        EventType.REFLECTION_PROPOSED,
        EventType.PERSONA_UPDATED,
        EventType.PERSONA_UPDATE_CLAMPED,
        EventType.PERSONA_UPDATE_REJECTED,
        EventType.CONSISTENCY_VIOLATION,
    }
    events = await store.get_events(sid, types=persona_types)
    persona_events = [e for e in events if e.payload.get("agent_id") == agent_id]
    return {
        "session_id": str(sid),
        "agent_id": agent_id,
        "persona": agent.persona if agent else None,
        "drift_mode": agent.drift.mode.value if agent else None,
        "current": current.model_dump(mode="json") if current else None,
        "timeline": [
            {
                "version": row.version,
                "state": row.state,
                "cause_seq": row.cause_seq,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in history
        ],
        "events": [e.model_dump(mode="json") for e in persona_events],
    }


def create_app(
    *,
    llm: LLMClient | None = None,
    database_url: str | None = None,
    settings: Settings | None = None,
    human_timeout_s: float = 60.0,
    memory: MemoryBackend | None = None,
    embedder: EmbeddingProvider | None = None,
) -> FastAPI:
    """Build the hub app (DESIGN.md §14). See module docstring for scope."""
    resolved_settings = settings or get_settings()
    resolved_db = database_url or resolved_settings.database_url
    resolved_llm = llm

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        engine = make_engine(resolved_db, load_vec=True)
        await init_db(engine)
        resolved_embedder = embedder or build_embedding_provider(
            backend=resolved_settings.embedding_backend,
            model=resolved_settings.embedding_model,
            dim=resolved_settings.embedding_dim,
        )
        await init_memory_tables(engine, resolved_embedder.dim, resolved_embedder.name)
        stored_model = await stored_embedding_model(engine)
        mismatch = embedding_mismatch_message(stored_model, resolved_embedder.name)
        if mismatch is not None:
            logger.warning(mismatch)
        resolved_memory = memory or SQLiteMemoryBackend(engine, resolved_embedder)
        factory = make_session_factory(engine)
        store = EventStore(factory)
        manager = SessionManager(
            store=store,
            llm=resolved_llm or _default_llm(resolved_settings),
            memory=resolved_memory,
            default_model=resolved_settings.default_model,
            human_timeout_s=human_timeout_s,
        )

        hub = WebHub(manager)
        app.state.deps = {
            "engine": engine,
            "factory": factory,
            "store": store,
            "manager": manager,
            "hub": hub,
            "settings": resolved_settings,
        }
        await manager.recover_on_start()
        await agent_repo.seed_defaults(factory)
        yield
        await hub.shutdown()
        await dispose(engine)

    app = FastAPI(title="Roundtable", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(_WEB / "static")), name="static")

    # ------------------------------------------------------------------ pages
    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        deps = _deps(request)
        store: EventStore = deps["store"]
        manager: SessionManager = deps["manager"]
        items = []
        for row in await store.list_sessions():
            state = await manager.get_state(uuid.UUID(row.id))
            items.append(
                {
                    "id": row.id,
                    "title": row.title,
                    "status": state.status,
                    "round": state.current_round,
                    "messages": state.message_count,
                    "cost": state.total_cost_usd,
                    "created_at": row.created_at,
                    "ended_reason": state.ended_reason,
                }
            )
        return _page("pages/dashboard.html", {"sessions": items, "active": "dashboard"})

    @app.get("/agents", response_class=HTMLResponse)
    async def agent_library(request: Request) -> HTMLResponse:
        deps = _deps(request)
        templates = await agent_repo.list_templates(deps["factory"])
        return _page(
            "pages/agents.html",
            {"agents": [agent_repo.template_summary(c) for c in templates], "active": "agents"},
        )

    @app.get("/agents/new", response_class=HTMLResponse)
    async def agent_form(request: Request) -> HTMLResponse:
        return _page("pages/agents_new.html", {"active": "agents"})

    @app.post("/agents")
    async def agent_create_from_form(request: Request) -> RedirectResponse:
        form = await request.form()
        try:
            config = _agent_from_form(form)
        except HTTPException:
            raise
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        factory = _deps(request)["factory"]
        try:
            await agent_repo.create_template(factory, config)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse("/agents", status_code=303)

    @app.get("/sessions/new", response_class=HTMLResponse)
    async def builder(request: Request) -> HTMLResponse:
        deps = _deps(request)
        templates = await agent_repo.list_templates(deps["factory"])
        session_templates = [
            {"id": tid, "title": cfg.title} for tid, cfg in await session_tpl.list_templates(deps["factory"])
        ]
        return _page(
            "pages/builder.html",
            {
                "agents": [agent_repo.template_summary(c) for c in templates],
                "strategies": sorted(load_strategies().keys()),
                "session_templates": session_templates,
                "active": "builder",
            },
        )

    @app.post("/sessions")
    async def create_from_builder(request: Request) -> RedirectResponse:
        form = await request.form()
        config = _config_from_form(form)
        deps = _deps(request)

        # "Mentés sablonként": store the config as a reusable session template.
        if (form.get("save_as_template") or "") == "1":
            template_id = str(form.get("template_id") or "").strip() or _slugify(config.title)
            if await session_tpl.update_template(deps["factory"], template_id, config) is None:
                await session_tpl.create_template(deps["factory"], template_id, config)
            return RedirectResponse("/sessions/new", status_code=303)

        entry = await _create_session(deps, config)
        deps["hub"].start(entry.session_id)
        if entry.humans:
            pid = next(iter(entry.humans))
            token = entry.tokens[pid]
            return RedirectResponse(f"/sessions/{entry.session_id}?participant={pid}&token={token}", status_code=303)
        return RedirectResponse(f"/sessions/{entry.session_id}", status_code=303)

    @app.get("/sessions/{session_id}", response_class=HTMLResponse)
    async def live_view(
        request: Request,
        session_id: str,
        participant: str | None = Query(default=None),
        token: str | None = Query(default=None),
    ) -> HTMLResponse:
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        deps = _deps(request)
        state = await deps["manager"].get_state(sid)
        human_id: str | None = None
        if participant is not None:
            hub: WebHub = deps["hub"]
            entry = hub.session(sid)
            if entry is None or entry.humans.get(participant) is None:
                raise HTTPException(status_code=404, detail="Unknown participant")
            if token is None or not _token_matches(entry.tokens, participant, token):
                raise HTTPException(status_code=403, detail="Invalid participant token")
            human_id = participant
        names = await _display_names(deps, state)
        approvals = [a for a in await deps["manager"].list_approvals(sid) if a["status"] == "pending"]
        return _page(
            "pages/live.html",
            {
                "state": state,
                "human_id": human_id,
                "sidebar": sidebar_fragment(state, names),
                "approvals": approvals,
            },
        )

    @app.get("/sessions/{session_id}/replay", response_class=HTMLResponse)
    async def replay_view(request: Request, session_id: str) -> HTMLResponse:
        """Event-by-event replay of a session (DESIGN.md §15, M6)."""
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        state = await _deps(request)["manager"].get_state(sid)
        return _page("pages/replay.html", {"state": state, "active": "live"})

    # ------------------------------------------------------------------- REST
    @app.get("/api/sessions")
    async def api_list_sessions(request: Request) -> list[dict[str, Any]]:
        deps = _deps(request)
        store: EventStore = deps["store"]
        manager: SessionManager = deps["manager"]
        out: list[dict[str, Any]] = []
        for row in await store.list_sessions():
            state = await manager.get_state(uuid.UUID(row.id))
            out.append(
                {
                    "id": row.id,
                    "title": row.title,
                    "status": state.status,
                    "current_round": state.current_round,
                    "message_count": state.message_count,
                    "total_cost_usd": state.total_cost_usd,
                    "ended_reason": state.ended_reason,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
            )
        return out

    @app.post("/api/sessions", status_code=201)
    async def api_create_session(body: SessionConfig, request: Request) -> dict[str, Any]:
        deps = _deps(request)
        entry = await _create_session(deps, body)
        return {
            "id": str(entry.session_id),
            "url": f"/sessions/{entry.session_id}",
            "humans": [
                {
                    "participant": pid,
                    "token": tok,
                    "url": f"/sessions/{entry.session_id}?participant={pid}&token={tok}",
                }
                for pid, tok in entry.tokens.items()
            ],
        }

    @app.get("/api/sessions/{session_id}")
    async def api_get_session(session_id: str, request: Request) -> dict[str, Any]:
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        manager: SessionManager = _deps(request)["manager"]
        data = (await manager.get_state(sid)).model_dump(mode="json")
        data["session_id"] = str(sid)
        return data

    async def _start_or_resume(request: Request, session_id: str) -> dict[str, Any]:
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        hub: WebHub = _deps(request)["hub"]
        if hub.start(sid):
            return {"status": "running"}
        if await hub.resume(sid):
            return {"status": "running"}
        raise HTTPException(status_code=409, detail="Session is not startable")

    app.post("/api/sessions/{session_id}/start")(_start_or_resume)
    app.post("/api/sessions/{session_id}/resume")(_start_or_resume)

    @app.post("/api/sessions/{session_id}/pause")
    async def api_pause(session_id: str, request: Request) -> dict[str, Any]:
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        hub: WebHub = _deps(request)["hub"]
        if not await hub.pause(sid):
            raise HTTPException(status_code=409, detail="Session is not running")
        return {"status": "paused"}

    @app.post("/api/sessions/{session_id}/stop")
    async def api_stop(session_id: str, request: Request) -> dict[str, Any]:
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        hub: WebHub = _deps(request)["hub"]
        if not hub.stop(sid):
            raise HTTPException(status_code=409, detail="Session is not active")
        return {"status": "stopping"}

    @app.get("/api/sessions/{session_id}/events")
    async def api_events(
        session_id: str,
        request: Request,
        after_seq: int = Query(default=0, ge=0),
        types: str | None = Query(default=None, description="Comma-separated event type names"),
    ) -> list[dict[str, Any]]:
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        store: EventStore = _deps(request)["store"]
        type_set: set[EventType] | None = None
        if types:
            try:
                type_set = {EventType(name) for name in types.split(",") if name}
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Bad event type") from exc
        events = await store.get_events(sid, after_seq=after_seq, types=type_set)
        return [e.model_dump(mode="json") for e in events]

    @app.get("/api/sessions/{session_id}/export")
    async def api_export(
        session_id: str,
        request: Request,
        format: str = Query(default="jsonl", pattern="^(jsonl|csv)$"),
    ) -> Response:
        """Analysis-ready export of the full event stream (DESIGN.md §14.1, M6)."""
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        store: EventStore = _deps(request)["store"]
        text = await store.export(sid, format=format)
        media_type = "application/x-ndjson" if format == "jsonl" else "text/csv"
        filename = f"{session_id}.{format}"
        return Response(
            content=text,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/sessions/{session_id}/approvals")
    async def api_approvals(session_id: str, request: Request) -> list[dict[str, Any]]:
        """List a session's persona-change approval requests (DESIGN.md §14.1, M6)."""
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        manager: SessionManager = _deps(request)["manager"]
        return await manager.list_approvals(sid)

    @app.post("/api/sessions/{session_id}/approvals/{agent_id}")
    async def api_decide_approval(
        session_id: str, agent_id: str, request: Request, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Approve or reject a pending persona change (DESIGN.md §12.3, M6)."""
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        decision = str(body.get("decision") or "").strip()
        if decision not in ("approve", "reject"):
            raise HTTPException(status_code=400, detail="decision must be 'approve' or 'reject'")
        manager: SessionManager = _deps(request)["manager"]
        try:
            decided = await manager.decide_approval(sid, agent_id, decision)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not decided:
            raise HTTPException(status_code=409, detail="No pending approval for that agent")
        return {"decision": decision, "agent_id": agent_id, "status": "ok"}

    @app.get("/sessions/{session_id}/agents/{agent_id}", response_class=HTMLResponse)
    async def agent_inspector(
        session_id: str, agent_id: str, request: Request
    ) -> HTMLResponse:
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        data = await _inspector_data(_deps(request), sid, agent_id)
        if data["persona"] is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        return _page("pages/inspector.html", {"data": data, "active": "live"})

    @app.get("/api/sessions/{session_id}/agents/{agent_id}")
    async def api_agent_state(
        session_id: str, agent_id: str, request: Request
    ) -> dict[str, Any]:
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        data = await _inspector_data(_deps(request), sid, agent_id)
        if data["persona"] is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        return data

    @app.get("/api/sessions/{session_id}/agents/{agent_id}/memory")
    async def api_search_memory(
        session_id: str,
        agent_id: str,
        request: Request,
        q: str = Query(default="", description="free-text query"),
        k: int = Query(default=5, ge=1, le=50),
        layer: str = Query(default="all", description="working | episodic | long_term | all"),
    ) -> dict[str, Any]:
        """Search an agent's memory (DESIGN.md §15 „memória-kereső")."""
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        factory = _deps(request)["factory"]
        if await agent_repo.get_template(factory, agent_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown agent {agent_id!r}")
        manager: SessionManager = _deps(request)["manager"]
        backend = manager.memory
        if backend is None:
            return {"query": q, "results": []}

        if layer == "all":
            layers = {Layer.EPISODIC, Layer.LONG_TERM}
        else:
            try:
                layers = {Layer(layer)}
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"unknown layer {layer!r}") from exc

        hits = await backend.search(agent_id, q, k, layers, session_id=sid)
        return {
            "query": q,
            "results": [
                {
                    "id": h.id,
                    "layer": h.layer.value,
                    "content": h.content,
                    "score": round(h.score, 4),
                    "source_seq": h.source_seq,
                }
                for h in hits
            ],
        }

    @app.get("/api/agents")
    async def api_agents(request: Request) -> list[dict[str, Any]]:
        templates = await agent_repo.list_templates(_deps(request)["factory"])
        return [agent_repo.template_summary(c) for c in templates]

    @app.post("/api/agents", status_code=201)
    async def api_create_agent(request: Request) -> dict[str, Any]:
        factory = _deps(request)["factory"]
        try:
            body = await request.json()
            config = AgentConfig.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            await agent_repo.create_template(factory, config)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return agent_repo.template_summary(config)

    @app.put("/api/agents/{agent_id}")
    async def api_update_agent(request: Request, agent_id: str) -> dict[str, Any]:
        factory = _deps(request)["factory"]
        try:
            body = await request.json()
            config = AgentConfig.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if config.id != agent_id:
            raise HTTPException(status_code=400, detail="id in path and body must match")
        updated = await agent_repo.update_template(factory, config)
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found")
        return agent_repo.template_summary(updated)

    @app.delete("/api/agents/{agent_id}", status_code=204)
    async def api_delete_agent(request: Request, agent_id: str) -> Response:
        factory = _deps(request)["factory"]
        if not await agent_repo.delete_template(factory, agent_id):
            raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found")
        return Response(status_code=204)

    @app.get("/api/session-templates")
    async def api_list_session_templates(request: Request) -> list[dict[str, Any]]:
        factory = _deps(request)["factory"]
        templates = await session_tpl.list_templates(factory)
        return [{"id": tid, "config": cfg.model_dump(mode="json")} for tid, cfg in templates]

    @app.get("/api/session-templates/{template_id}")
    async def api_get_session_template(request: Request, template_id: str) -> dict[str, Any]:
        factory = _deps(request)["factory"]
        config = await session_tpl.get_template(factory, template_id)
        if config is None:
            raise HTTPException(status_code=404, detail=f"Session template {template_id!r} not found")
        return config.model_dump(mode="json")

    @app.post("/api/session-templates", status_code=201)
    async def api_save_session_template(request: Request) -> dict[str, Any]:
        factory = _deps(request)["factory"]
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="body must be JSON") from exc
        template_id = str(body.get("id") or "").strip()
        if not template_id:
            raise HTTPException(status_code=400, detail="id is required")
        try:
            config = SessionConfig.model_validate(body.get("config") or {})
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if await session_tpl.update_template(factory, template_id, config) is None:
            await session_tpl.create_template(factory, template_id, config)
        return {"id": template_id, "config": config.model_dump(mode="json")}

    @app.delete("/api/session-templates/{template_id}", status_code=204)
    async def api_delete_session_template(request: Request, template_id: str) -> Response:
        factory = _deps(request)["factory"]
        if not await session_tpl.delete_template(factory, template_id):
            raise HTTPException(status_code=404, detail=f"Session template {template_id!r} not found")
        return Response(status_code=204)

    @app.get("/api/plugins")
    async def api_plugins() -> dict[str, Any]:
        strategies = load_strategies()
        return {
            "turn_strategies": [
                {
                    "name": cls.name,
                    "class": f"{cls.__module__}.{cls.__qualname__}",
                    "params_schema": cls.Params.model_json_schema(),
                }
                for cls in strategies.values()
            ]
        }

    # ---------------------------------------------------------------- WebSocket
    @app.websocket("/ws/sessions/{session_id}")
    async def ws_session(
        ws: WebSocket,
        session_id: str,
        participant: str | None = Query(default=None),
        token: str | None = Query(default=None),
        after_seq: int = Query(default=0, ge=0),
    ) -> None:
        await ws.accept()
        sid = _parse_session_id(session_id)
        if sid is None:
            await ws.close(code=4404)
            return
        deps = _deps(ws)
        hub: WebHub = deps["hub"]
        manager: SessionManager = deps["manager"]
        store: EventStore = deps["store"]

        human_id: str | None = None
        if participant is not None:
            if hub.resolve_token(sid, token or "") != participant:
                await ws.close(code=4401)
                return
            human_id = participant
        ws.roundtable_participant = human_id  # type: ignore[attr-defined]

        # Replay from the database, applying each event to a local state so the
        # fragments render exactly as they would live (DESIGN.md §14.2).
        state = SessionState()
        history = await store.get_events(sid)
        for event in history:
            apply_event(state, event)
        names = await _display_names(deps, state)
        await ws.send_json(
            {"type": "hello", "session_id": str(sid), "last_seq": len(history), "participant": human_id}
        )
        for event in history:
            if event.seq > after_seq:
                await ws.send_json(await _event_message(event, state, names))

        # If this human's turn is already pending (assigned but not yet
        # answered) when they (re)connect, hand them the prompt now — the live
        # path does the same from the bus (DESIGN.md §14.2 ``your_turn``).
        if human_id is not None and state.status == "running" and _pending_turn_for(history, human_id):
            deadline = datetime.now(UTC) + timedelta(seconds=manager.human_timeout_s)
            await ws.send_json(
                {"type": "your_turn", "participant": human_id, "deadline": deadline.isoformat()}
            )

        # Live tail through the bus.
        humans = hub.session(sid)
        human_ids = set(humans.humans) if humans else set()

        async def on_event(event: Event) -> None:
            apply_event(state, event)
            if event.type is EventType.TURN_ASSIGNED:
                speaker = str(event.payload.get("speaker_id", ""))
                if speaker in human_ids:
                    deadline = datetime.now(UTC) + timedelta(seconds=manager.human_timeout_s)
                    await hub.send_to(
                        sid,
                        speaker,
                        {"type": "your_turn", "participant": speaker, "deadline": deadline.isoformat()},
                    )
            await hub.broadcast(sid, await _event_message(event, state, names))

        unsubscribe = manager.subscribe(sid, on_event)
        hub.add_ws(sid, ws)
        try:
            while True:
                raw = await ws.receive_text()
                await _handle_client_message(hub, sid, human_id, raw)
        except WebSocketDisconnect:
            pass
        finally:
            unsubscribe()
            hub.drop_ws(sid, ws)

    return app


def _token_matches(tokens: dict[str, str], participant: str, presented: str) -> bool:
    import secrets

    expected = tokens.get(participant)
    return expected is not None and secrets.compare_digest(expected, presented)

def _pending_turn_for(history: list[Event], participant_id: str) -> bool:
    """True if the last turn assigned to ``participant_id`` is still unanswered."""
    pending = False
    for event in history:
        if event.type is EventType.TURN_ASSIGNED and event.payload.get("speaker_id") == participant_id:
            pending = True
        elif event.type in (EventType.MESSAGE_POSTED, EventType.TURN_SKIPPED):
            speaker = event.payload.get("speaker_id") or (event.payload.get("message") or {}).get("speaker_id")
            if speaker == participant_id:
                pending = False
    return pending

async def _display_names(deps: dict[str, Any], state: SessionState) -> dict[str, str]:
    """Map participant id -> display name, preferring agent persona names."""
    names = {ref.participant_id: ref.participant_id for ref in (state.config.participants if state.config else [])}
    if not names:
        return names
    templates = await agent_repo.list_templates(deps["factory"])
    by_id = {t.id: t for t in templates}
    for pid in list(names):
        template = by_id.get(pid)
        if template is not None:
            names[pid] = template.persona.name
    return names


async def _event_message(event: Event, state: SessionState, names: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "event",
        "seq": event.seq,
        "event": event.model_dump(mode="json"),
        "fragments": {
            "transcript": transcript_fragment(event, state, names),
            "sidebar": sidebar_fragment(state, names),
        },
    }


async def _handle_client_message(hub: WebHub, sid: uuid.UUID, human_id: str | None, raw: str) -> None:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return
    if human_id is None:
        return
    mtype = msg.get("type")
    if mtype == "say":
        human = hub.human(sid, human_id)
        if human is not None:
            human.submit(str(msg.get("content", "")))
    elif mtype == "raise_hand":
        engine = hub.engine(sid)
        if engine is not None:
            await engine.raise_hand(human_id)
    elif mtype == "lower_hand":
        engine = hub.engine(sid)
        human = hub.human(sid, human_id)
        if engine is not None and human is not None:
            human.lower_hand()
            await engine.emit(EventType.HAND_LOWERED, participant_id=human_id)


def asgi_factory() -> FastAPI:
    """Zero-arg app factory for ``uvicorn ... --factory`` (used by the CLI)."""
    return create_app()
