"""FastAPI hub application.

Wires the M1 core (manager / store / engine) to HTTP: REST for control and
data, one WebSocket per session for the lossless live feed (replay from
``after_seq`` then live), and Jinja2+HTMX pages for the boardroom, persona
templates, session builder, and the live table.

The app is built by :func:`create_app` so tests (and the CLI) can inject a
:class:`~thinktank.llm.client.LLMClient` (e.g. ``FakeLLM``) and a throwaway
database.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.background import BackgroundTask
from starlette.datastructures import FormData

from thinktank.config import Settings, get_settings, reload_settings, write_env_settings
from thinktank.core.manager import SessionManager
from thinktank.core.state import STATUS_CREATED, SessionState, apply_event
from thinktank.domain.events import Event, EventType
from thinktank.domain.models import (
    AgentConfig,
    DriftMode,
    Layer,
    PersonaCore,
    RemoteConfig,
    SessionConfig,
)
from thinktank.llm.client import LLMClient
from thinktank.llm.providers import (
    DEFAULT_BASE_URLS,
    PROVIDER_FAMILIES,
    PROVIDER_TYPES,
    LLMProvider,
    ProviderRegistry,
)
from thinktank.memory.base import MemoryBackend
from thinktank.memory.embeddings import EmbeddingProvider, build_embedding_provider
from thinktank.memory.sqlite import SQLiteMemoryBackend
from thinktank.plugins.registry import load_strategies
from thinktank.storage.db import (
    dispose,
    init_db,
    init_memory_tables,
    make_engine,
    make_session_factory,
    stored_embedding_model,
)
from thinktank.storage.repositories import EventStore
from thinktank.web import agents as agent_repo
from thinktank.web import session_templates as session_tpl
from thinktank.web.hub import HubSession, WebHub
from thinktank.web.render import sidebar_fragment, transcript_fragment

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
    context = {**context, "title": context.get("title", "ThinkTank")}
    html = env.get_template(template).render(
        **context,
        static_version=_static_version(),
    )
    return HTMLResponse(html)


def _static_version() -> str:
    """Cache-busting token derived from the static assets' mtimes.

    Any change to ``live.js``/``app.css``/… produces a new URL, so a browser
    can never serve a stale copy from its heuristic cache.
    """
    latest = 0
    for entry in (_WEB / "static").iterdir():
        if entry.is_file():
            latest = max(latest, int(entry.stat().st_mtime))
    return str(latest)


def embedding_mismatch_message(stored: str | None, configured: str) -> str | None:
    """Warn when stored memory vectors predate the configured embedding model.

    ``None`` means no mismatch (nothing stored yet, or the model already matches).
    """
    if stored is None or stored == configured:
        return None
    return (
        f"embedding model mismatch: stored memory vectors use {stored!r} but the configured "
        f"provider is {configured!r}. Retrieval quality may be degraded; run "
        f"`thinktank reembed` to recompute the vectors."
    )


def _default_llm(settings: Settings) -> LLMClient:
    if settings.fake_llm:
        from thinktank.llm.fake import FakeLLM, sample_responses

        return FakeLLM(responses=sample_responses(60))
    from thinktank.llm.litellm_client import LiteLLMClient

    return LiteLLMClient(timeout_s=settings.llm_timeout_s)


async def _create_session(
    deps: dict[str, Any],
    config: SessionConfig,
    remotes: dict[str, RemoteConfig] | None = None,
) -> HubSession:
    """Validate, resolve agent templates, and register the session in the hub."""
    agent_ids = config.agent_ids()
    try:
        agents = await agent_repo.resolve_agents(deps["factory"], agent_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await deps["hub"].create_session(config, agents, remotes)


def _parse_remotes(raw: Any) -> dict[str, RemoteConfig] | None:
    """Parse an optional top-level ``remotes`` object from a raw request body."""
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="'remotes' must be an object keyed by participant id")
    try:
        return {rid: RemoteConfig.model_validate({"id": rid, **rcfg}) for rid, rcfg in raw.items()}
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _rehydrate_created_sessions(store: EventStore, factory: Any, hub: WebHub) -> None:
    """Re-attach a live engine for every stored session still in ``created`` status.

    The hub's in-memory session registry starts empty on every process
    restart; without this, a session that was created but never started would
    be stuck returning 409 from ``/start`` forever.
    """
    for row in await store.list_sessions():
        if row.status != STATUS_CREATED:
            continue
        config = SessionConfig.model_validate(row.config)
        try:
            agents = await agent_repo.resolve_agents(factory, config.agent_ids())
            await hub.load_created(uuid.UUID(row.id), config, agents)
        except Exception:
            logger.exception("failed to rehydrate created session %s", row.id)


def _multi(form: FormData, key: str) -> list[str]:
    """Normalize a possibly multi-valued form field to a list of trimmed strings."""
    values = form.getlist(key)
    return [value.strip() for value in values if isinstance(value, str) and value.strip()]


async def _fetch_provider_models(provider_family: str, base_url: str | None, api_key: str | None) -> list[str]:
    """List the models actually offered by an endpoint (Settings → LLM providers).

    Raises ``ValueError`` with a user-facing message when the family
    cannot be listed or the request fails.
    """

    async def get_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            raise ValueError(f"unavailable: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("unexpected response from the endpoint")
        return data

    if provider_family == "azure":
        raise ValueError(
            "Azure: models cannot be listed (tenant-specific deployments) — enter them in the 'Other' models field"
        )
    if provider_family == "bedrock":
        raise ValueError("Bedrock: models cannot be listed (AWS auth) — enter them in the 'Other' models field")
    if provider_family == "ollama":
        if not base_url:
            raise ValueError("A base URL is required (e.g. http://localhost:11434)")
        data = await get_json(base_url.rstrip("/") + "/api/tags", {})
        return sorted(m.get("name") for m in data.get("models", []) if m.get("name"))
    if provider_family == "anthropic":
        if not api_key:
            raise ValueError("An API key is required")
        data = await get_json(
            "https://api.anthropic.com/v1/models",
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )
        return sorted(m.get("id") for m in data.get("data", []) if m.get("id"))
    if provider_family == "gemini":
        if not api_key:
            raise ValueError("An API key is required")
        data = await get_json(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}", {})
        names = []
        for m in data.get("models", []):
            name = m.get("modelId") or (m.get("name") or "").removeprefix("models/")
            if name:
                names.append(name)
        return sorted(names)
    # openai, openrouter, mistral, groq: OpenAI-compatible GET <base>/models
    url = (base_url or DEFAULT_BASE_URLS.get(provider_family) or "https://api.openai.com/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = await get_json(url + "/models", headers)
    return sorted(m.get("id") for m in data.get("data", []) if m.get("id"))

def _config_from_form(form: FormData) -> SessionConfig:
    """Build a SessionConfig from the builder form."""
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
        existing_ids = {p.get("agent") or p.get("human") or p.get("remote") for p in participants}
        if moderator not in existing_ids:
            participants = [{"agent": moderator}, *participants]

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
    """Build an AgentConfig from the library form."""

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
        age=str(form.get("persona_age") or "").strip(),
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
    """Assemble the agent-inspector payload (M4)."""
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


# Settings form ("Settings"): one entry per Settings field.
# ``live`` = the value is re-read on the next use without a hub restart.
SETTINGS_SECTIONS: list[dict[str, Any]] = [
    {
        "title": "LLM",
        "fields": [
            {"name": "default_model", "label": "Default model", "type": "select", "live": False,
             "hint": "Registered providers' models only (Settings → LLM providers); used when an agent sets none"},
            {"name": "fake_llm", "label": "Offline FakeLLM (no API calls)", "type": "bool", "live": False},
            {"name": "llm_timeout_s", "label": "LLM timeout (s)", "type": "number", "live": True},
            {"name": "llm_max_concurrency", "label": "Max. concurrent LLM calls", "type": "number", "live": True},
        ],
    },
    {
        "title": "Memory / embedding",
        "fields": [
            {"name": "embedding_backend", "label": "Embedding backend", "type": "select",
             "options": ["fake", "litellm"], "live": False},
            {"name": "embedding_model", "label": "Embedding model", "type": "text", "live": True,
             "hint": "for the litellm backend; ignored with the fake backend"},
            {"name": "embedding_dim", "label": "Embedding dimension", "type": "number", "live": False,
             "hint": "must match the model's output size; after a change: uv run thinktank reembed"},
        ],
    },
    {
        "title": "Hub",
        "fields": [
            {"name": "host", "label": "Host", "type": "text", "live": False,
             "hint": "in v1, only 127.0.0.1 is recommended"},
            {"name": "port", "label": "Port", "type": "number", "live": False},
        ],
    },
    {
        "title": "Storage",
        "fields": [
            {"name": "database_url", "label": "SQLite URL", "type": "text", "live": False},
            {"name": "data_dir", "label": "Data directory (relative SQLite paths)", "type": "text", "live": False},
        ],
    },
    {
        "title": "Logging",
        "fields": [
            {"name": "log_level", "label": "Log level", "type": "select",
             "options": ["DEBUG", "INFO", "WARNING", "ERROR"], "live": False},
            {"name": "log_json", "label": "JSON-formatted logs", "type": "bool", "live": False},
        ],
    },
]


_RESTART_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ThinkTank — restarting</title>
<style>
  body {
    font-family: -apple-system, system-ui, sans-serif;
    background: #f5f4f0; color: #222;
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh; margin: 0;
  }
  .card {
    background: #fff; border: 1px solid #ddd; border-radius: 12px;
    padding: 32px 40px; max-width: 440px; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
  }
  h1 { font-size: 20px; margin: 0 0 12px; }
  p { margin: 0 0 16px; color: #555; font-size: 14px; }
  a { color: #2b5d8a; }
</style>
</head>
<body>
  <div class="card">
    <h1>The hub is restarting…</h1>
    <p>Settings saved to the <code>.env</code> file.
    When the hub is reachable again, we open the Settings page.</p>
    <a href="/settings?saved=1">Open Settings</a>
  </div>
  <script>
    const go = async () => {
      try {
        const r = await fetch("/settings", { cache: "no-store" });
        if (r.ok) { location.replace("/settings?saved=1"); return; }
      } catch (e) { /* the hub is still restarting */ }
      setTimeout(go, 1000);
    };
    setTimeout(go, 1500);
  </script>
</body>
</html>
"""


def _parse_settings_form(form: FormData) -> dict[str, Any]:
    """Map the settings form onto candidate ``Settings`` values (400 on bad numbers)."""
    data: dict[str, Any] = {}
    errors: list[str] = []
    for section in SETTINGS_SECTIONS:
        for f in section["fields"]:
            name = f["name"]
            raw = form.get(name)
            if f["type"] == "bool":
                data[name] = raw == "on"
            elif f["type"] == "number":
                try:
                    text = str(raw or "").strip()
                    data[name] = float(text) if "." in text else int(text)
                except ValueError:
                    errors.append(f"{f['label']} must be a number")
            else:
                value = str(raw or "").strip()
                data[name] = value if value else None
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))
    return data


def _save_settings_data(data: dict[str, Any]) -> None:
    """Validate the candidate settings, then persist them into the ``.env`` file."""
    try:
        Settings(_env_file=None, **data)  # pyright: ignore[reportCallIssue] - pydantic-settings private init kwarg
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    env_values = {
        name: (
            "true" if value is True else "false" if value is False else "" if value is None else str(value)
        )
        for name, value in data.items()
    }
    write_env_settings(Path(".env"), env_values)
    reload_settings()


def _self_restart() -> None:
    """Exit so the supervisor (the ``start.sh`` loop) relaunches us with the new settings."""
    time.sleep(0.5)
    os.kill(os.getpid(), signal.SIGTERM)


def create_app(
    *,
    llm: LLMClient | None = None,
    database_url: str | None = None,
    settings: Settings | None = None,
    human_timeout_s: float = 60.0,
    memory: MemoryBackend | None = None,
    embedder: EmbeddingProvider | None = None,
) -> FastAPI:
    """Build the hub app. See module docstring for scope."""
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
        await agent_repo.ensure_colors(factory)
        await _rehydrate_created_sessions(store, factory, hub)
        yield
        await hub.shutdown()
        await dispose(engine)

    app = FastAPI(title="ThinkTank", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(_WEB / "static")), name="static")

    @app.middleware("http")
    async def static_no_cache(
        request: Request, call_next: Callable[[Request], Coroutine[Any, Any, Response]]
    ) -> Response:
        """Revalidate static assets on every load (single-machine tool).

        Without a ``Cache-Control`` header browsers apply heuristic caching and
        can serve a stale ``live.js``/``app.js`` after a code change; ETag
        revalidation is cheap on a local single-machine deployment.
        """
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

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
        return _page("pages/dashboard.html", {"sessions": items, "active": "dashboard", "title": "Boardroom"})

    # LLM providers registered from the Settings page.
    provider_registry = ProviderRegistry(Path(resolved_settings.data_dir) / "llm_providers.json")

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request) -> HTMLResponse:
        """All Settings entries as an editable form (mirrors the ``.env`` file)."""
        settings = get_settings()
        values = {
            f["name"]: getattr(settings, f["name"])
            for section in SETTINGS_SECTIONS
            for f in section["fields"]
        }
        saved_flag = request.query_params.get("saved") == "1"
        providers = provider_registry.list()
        model_options: list[str | dict[str, str]] = [*provider_registry.model_options()]
        if settings.default_model and settings.default_model not in model_options:
            model_options.append(
                {"value": settings.default_model, "label": f"{settings.default_model} (current)"}
            )
        sections = [
            {
                "title": section["title"],
                "fields": [
                    {**f, "options": model_options} if f["name"] == "default_model" else {**f}
                    for f in section["fields"]
                ],
            }
            for section in SETTINGS_SECTIONS
        ]
        edit_id = (request.query_params.get("edit") or "").strip()
        edit_provider = next((p for p in providers if p.id == edit_id), None)
        return _page(
            "pages/settings.html",
            {
                "sections": sections,
                "values": values,
                "saved": saved_flag,
                "providers": providers,
                "edit_provider": edit_provider,
                "provider_types": PROVIDER_TYPES,
                "provider_defaults": DEFAULT_BASE_URLS,
                "active": "settings",
                "title": "Settings",
            },
        )

    @app.post("/settings")
    async def settings_save(request: Request) -> RedirectResponse:
        """Validate the form, write the THINKTANK_ lines into ``.env``, re-read settings."""
        _save_settings_data(_parse_settings_form(await request.form()))
        return RedirectResponse("/settings?saved=1", status_code=303)

    @app.post("/settings/restart")
    async def settings_save_restart(request: Request) -> Response:
        """Save the settings, then restart the hub so the new values take effect now."""
        _save_settings_data(_parse_settings_form(await request.form()))
        response = HTMLResponse(_RESTART_PAGE)
        response.background = BackgroundTask(_self_restart)
        return response

    @app.post("/settings/providers")
    async def provider_save(request: Request) -> RedirectResponse:
        """Register (or update) an LLM provider endpoint."""
        form = await request.form()

        def opt(key: str) -> str | None:
            value = str(form.get(key) or "").strip()
            return value or None

        base_url = opt("base_url")
        if base_url is not None and not base_url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="Base URL must start with http(s)://")
        provider_id = (opt("id") or "").lower()
        if provider_id and not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", provider_id):
            raise HTTPException(
                status_code=400,
                detail=(
                    "ID may only contain lowercase letters, digits, '-' or '_', must start with a letter or digit, "
                    "max. 32 characters (e.g. owentest)"
                ),
            )
        models = [m.strip() for m in str(form.get("models") or "").split(",") if m.strip()]
        existing = next((p for p in provider_registry.list() if p.id == provider_id), None)
        api_key = opt("api_key")
        if api_key is None and existing is not None:
            api_key = existing.api_key
        try:
            provider = LLMProvider(
                id=provider_id,
                provider=opt("provider") or "openai",
                base_url=base_url,
                api_key=api_key,
                models=models,
            )
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if provider.provider not in PROVIDER_FAMILIES:
            raise HTTPException(status_code=400, detail=f"unknown provider type: {provider.provider}")
        provider_registry.upsert(provider)
        return RedirectResponse("/settings?saved=1", status_code=303)

    @app.post("/settings/providers/delete")
    async def provider_delete(request: Request) -> RedirectResponse:
        """Remove a registered LLM provider."""
        form = await request.form()
        provider_id = str(form.get("id") or "").strip()
        if not provider_registry.remove(provider_id):
            raise HTTPException(status_code=404, detail=f"no such provider: {provider_id or '?'}")
        return RedirectResponse("/settings?saved=1", status_code=303)

    @app.post("/api/provider-models")
    async def provider_models_list(request: Request) -> JSONResponse:
        """List the models actually available at a provider endpoint (the provider form's Refresh button)."""
        body = await request.json()
        provider_id = str(body.get("id") or "").strip()
        if provider_id:
            p = next((x for x in provider_registry.list() if x.id == provider_id), None)
            if p is None:
                raise HTTPException(status_code=404, detail=f"no such provider: {provider_id}")
            family, base_url, api_key = p.provider, p.base_url, p.api_key
        else:
            family = str(body.get("provider") or "openai")
            base_url = (str(body.get("base_url") or "").strip()) or None
            api_key = (str(body.get("api_key") or "").strip()) or None
        if family not in PROVIDER_FAMILIES:
            raise HTTPException(status_code=400, detail=f"unknown provider type: {family}")
        try:
            models = await _fetch_provider_models(family, base_url, api_key)
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return JSONResponse({"models": models})

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
        return _page(
            "pages/agent_form.html",
            {"agent": None, "model_options": provider_registry.model_options(), "active": "agents"},
        )

    @app.get("/agents/{agent_id}/edit", response_class=HTMLResponse)
    async def agent_edit(agent_id: str, request: Request) -> HTMLResponse:
        factory = _deps(request)["factory"]
        config = await agent_repo.get_template(factory, agent_id)
        if config is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found")
        return _page(
            "pages/agent_form.html",
            {
                "agent": agent_repo.template_summary(config),
                "model_options": provider_registry.model_options(),
                "active": "agents",
            },
        )

    @app.post("/agents/{agent_id}")
    async def agent_update_from_form(agent_id: str, request: Request) -> RedirectResponse:
        form = await request.form()
        factory = _deps(request)["factory"]
        if await agent_repo.get_template(factory, agent_id) is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found")
        try:
            config = _agent_from_form(form)
        except HTTPException:
            raise
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if config.id != agent_id:
            raise HTTPException(status_code=400, detail="id in path and form must match")
        await agent_repo.update_template(factory, config)
        return RedirectResponse("/agents", status_code=303)

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

        # "Save as template": store the config as a reusable session template.
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

    @app.get("/sessions/{session_id}/edit", response_class=HTMLResponse)
    async def edit_session_page(request: Request, session_id: str) -> HTMLResponse:
        """Prefilled config form for a session that has not started yet."""
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        deps = _deps(request)
        row = await deps["store"].get_session(sid)
        if row is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        config = SessionConfig.model_validate(row.config)
        templates = await agent_repo.list_templates(deps["factory"])
        session_templates = [
            {"id": tid, "title": cfg.title} for tid, cfg in await session_tpl.list_templates(deps["factory"])
        ]
        return _page(
            "pages/builder.html",
            {
                "config": config,
                "form_action": f"/sessions/{sid}/edit",
                "agents": [agent_repo.template_summary(c) for c in templates],
                "strategies": sorted(load_strategies().keys()),
                "session_templates": session_templates,
                "active": "builder",
            },
        )

    @app.post("/sessions/{session_id}/edit")
    async def save_session_edit(request: Request, session_id: str) -> RedirectResponse:
        """Save an edited config. Only allowed while the session is ``created``
        (fresh or reset) — once the engine ran, the conversation owns the state."""
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        deps = _deps(request)
        row = await deps["store"].get_session(sid)
        if row is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        if row.status != "created":
            raise HTTPException(status_code=409, detail="Edit only before the session starts (Reset it first)")
        form = await request.form()
        config = _config_from_form(form)
        try:
            agents = await agent_repo.resolve_agents(deps["factory"], config.agent_ids())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await deps["store"].update_session_config(sid, config)
        if (form.get("save_as_template") or "") == "1":
            template_id = str(form.get("template_id") or "").strip() or _slugify(config.title)
            if await session_tpl.update_template(deps["factory"], template_id, config) is None:
                await session_tpl.create_template(deps["factory"], template_id, config)
        hub: WebHub = deps["hub"]
        entry = hub.session(sid)
        if entry is not None and entry.task is None:
            # Rebuild the not-yet-started engine so it holds the new config
            # (wipes nothing of consequence: a ``created`` session has no
            # conversation yet, only its SESSION_CREATED event).
            await hub.reset(sid, config, agents)
        return RedirectResponse(f"/sessions/{sid}", status_code=303)

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
        if await deps["store"].get_session(sid) is None:
            raise HTTPException(status_code=404, detail="Unknown session")
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
        names, colors = await _participant_meta(deps, state)
        approvals = [a for a in await deps["manager"].list_approvals(sid) if a["status"] == "pending"]
        return _page(
            "pages/live.html",
            {
                "state": state,
                "human_id": human_id,
                "sidebar": sidebar_fragment(state, names, colors),
                "approvals": approvals,
            },
        )

    @app.get("/sessions/{session_id}/download")
    async def download_session(request: Request, session_id: str) -> Response:
        """Download the full session (config + event stream) as one JSON file."""
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        deps = _deps(request)
        store: EventStore = deps["store"]
        if await store.get_session(sid) is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        state = await deps["manager"].get_state(sid)
        events = await store.get_events(sid)
        payload = {
            "session_id": str(sid),
            "config": state.config.model_dump(mode="json") if state.config else None,
            "status": state.status,
            "current_round": state.current_round,
            "message_count": state.message_count,
            "total_cost_usd": state.total_cost_usd,
            "started_at": state.started_at.isoformat() if state.started_at else None,
            "ended_at": state.ended_at.isoformat() if state.ended_at else None,
            "events": [e.model_dump(mode="json") for e in sorted(events, key=lambda ev: ev.seq)],
        }
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        return Response(
            content=body,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="session-{session_id}.json"'},
        )

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
        remotes = _parse_remotes((await request.json()).get("remotes"))
        entry = await _create_session(deps, body, remotes)
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
        deps = _deps(request)
        if await deps["store"].get_session(sid) is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        manager: SessionManager = deps["manager"]
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

    @app.post("/api/sessions/{session_id}/reset")
    async def api_reset(session_id: str, request: Request) -> dict[str, Any]:
        """Wipe the conversation and restore the session to its initial state.

        The session keeps its id, config, and human tokens; the event stream
        starts over from ``SESSION_CREATED`` and the session can be started
        again with ``/start`` (control bar).
        """
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        deps = _deps(request)
        row = await deps["store"].get_session(sid)
        if row is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        config = SessionConfig.model_validate(row.config)
        try:
            agents = await agent_repo.resolve_agents(deps["factory"], config.agent_ids())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        hub: WebHub = deps["hub"]
        await hub.reset(sid, config, agents)
        return {"id": str(sid), "status": "created"}

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def api_delete_session(session_id: str, request: Request) -> Response:
        """Delete a session and its stored conversation (control bar).

        The session row, events, messages, persona history and approvals are
        removed; agent memory is kept. Refused while the session is active.
        """
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        deps = _deps(request)
        store: EventStore = deps["store"]
        if await store.get_session(sid) is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        state = await deps["manager"].get_state(sid)
        if state.status in ("running", "paused"):
            raise HTTPException(status_code=409, detail="Stop the session before deleting it")
        hub: WebHub = deps["hub"]
        await hub.delete(sid)
        if not await store.delete_session(sid):
            raise HTTPException(status_code=404, detail="Unknown session")
        return Response(status_code=204)

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
        """Analysis-ready export of the full event stream (M6)."""
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
        """List a session's persona-change approval requests (M6)."""
        sid = _parse_session_id(session_id)
        if sid is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        manager: SessionManager = _deps(request)["manager"]
        return await manager.list_approvals(sid)

    @app.post("/api/sessions/{session_id}/approvals/{agent_id}")
    async def api_decide_approval(
        session_id: str, agent_id: str, request: Request, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Approve or reject a pending persona change (M6)."""
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
        return _page("pages/inspector.html", {"data": data, "active": "live", "title": "Agent inspector"})

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
        """Search an agent's memory ("memory search")."""
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
            created = await agent_repo.create_template(factory, config)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return agent_repo.template_summary(created)

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
        ws.thinktank_participant = human_id  # type: ignore[attr-defined]

        # Replay from the database, applying each event to a local state so the
        # fragments render exactly as they would live.
        state = SessionState()
        history = await store.get_events(sid)
        for event in history:
            apply_event(state, event)
        names, colors = await _participant_meta(deps, state)
        await ws.send_json(
            {
                "type": "hello",
                "session_id": str(sid),
                "last_seq": len(history),
                "status": state.status,
                "participant": human_id,
            }
        )
        for event in history:
            if event.seq > after_seq:
                await ws.send_json(await _event_message(event, state, names, colors))

        # If this human's turn is already pending (assigned but not yet
        # answered) when they (re)connect, hand them the prompt now — the live
        # path does the same from the bus (``your_turn``).
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
            await hub.broadcast(sid, await _event_message(event, state, names, colors))

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

async def _participant_meta(deps: dict[str, Any], state: SessionState) -> tuple[dict[str, str], dict[str, str]]:
    """Map participant id -> display name and -> display color (agents only)."""
    names = {ref.participant_id: ref.participant_id for ref in (state.config.participants if state.config else [])}
    if not names:
        return names, {}
    templates = await agent_repo.list_templates(deps["factory"])
    by_id = {t.id: t for t in templates}
    colors: dict[str, str] = {}
    for pid in list(names):
        template = by_id.get(pid)
        if template is not None:
            names[pid] = template.persona.name
            if template.color:
                colors[pid] = template.color
    return names, colors


async def _event_message(
    event: Event, state: SessionState, names: dict[str, str], colors: dict[str, str]
) -> dict[str, Any]:
    return {
        "type": "event",
        "seq": event.seq,
        "event": event.model_dump(mode="json"),
        "fragments": {
            "transcript": transcript_fragment(event, state, names, colors),
            "sidebar": sidebar_fragment(state, names, colors),
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
        human = hub.human(sid, human_id)
        if engine is not None and human is not None:
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
