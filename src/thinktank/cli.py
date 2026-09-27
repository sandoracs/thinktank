"""Command-line interface (DESIGN.md §17, §18 M1).

``thinktank run template.yaml`` runs a full debate headless — the design's
"usable for research from the CLI without the UI" milestone (DESIGN.md §18).
Every posted message is printed live as it happens.

``--fake`` runs the whole pipeline against the offline :class:`FakeLLM`, so the
system can be exercised end-to-end with no API key or model download.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from thinktank.config import get_settings
from thinktank.core.manager import SessionManager
from thinktank.domain.events import Event, EventType, MessagePostedPayload
from thinktank.domain.models import AgentConfig, SessionConfig
from thinktank.llm.client import LLMClient
from thinktank.memory.embeddings import build_embedding_provider
from thinktank.memory.sqlite import SQLiteMemoryBackend
from thinktank.storage.db import dispose, init_db, init_memory_tables, make_engine, make_session_factory
from thinktank.storage.repositories import EventStore

logger = logging.getLogger("thinktank")




def _load_session(path: Path) -> tuple[SessionConfig, dict[str, AgentConfig]]:
    import yaml

    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    agents: dict[str, AgentConfig] = {}
    for aid, acfg in (data.get("agents") or {}).items():
        agents[aid] = AgentConfig(id=aid, **(acfg or {}))
    session_data = {k: v for k, v in data.items() if k != "agents"}
    return SessionConfig(**session_data), agents


def _build_llm(fake: bool) -> LLMClient:
    if fake:
        from thinktank.llm.fake import FakeLLM, sample_responses

        # A generous pool so a multi-round fake run does not repeat immediately.
        return FakeLLM(responses=sample_responses(60))
    from thinktank.llm.litellm_client import LiteLLMClient

    return LiteLLMClient()


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    config, agents = _load_session(args.session)

    engine = make_engine(load_vec=True)
    await init_db(engine)
    embedder = build_embedding_provider(
        backend=settings.embedding_backend,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
    )
    await init_memory_tables(engine, embedder.dim, embedder.name)
    factory = make_session_factory(engine)
    store = EventStore(factory)
    manager = SessionManager(
        store=store,
        llm=_build_llm(args.fake),
        memory=SQLiteMemoryBackend(engine, embedder),
        default_model=settings.default_model,
        human_timeout_s=args.human_timeout,
    )
    await manager.recover_on_start()

    session = await manager.create_session(config, agents)
    session_id = session.state.session_id
    assert session_id is not None

    def _print_event(event: Event) -> None:
        if event.type is EventType.MESSAGE_POSTED:
            message = event.payload_as(MessagePostedPayload).message
            print(f"\n[{event.seq}] {message.speaker_id} ({message.kind}):\n{message.content}\n")
        elif event.type is EventType.ROUND_STARTED:
            print(f"\n=== Round {event.payload.get('round')} ===")
        elif event.type is EventType.SESSION_ENDED:
            print(f"\n=== Session ended: {event.payload.get('reason')} ===")
        elif event.type is EventType.MEMORY_WRITTEN:
            payload = event.payload
            print(f"\n[memory] {payload.get('agent_id')} wrote {payload.get('layer')} (id={payload.get('memory_id')})")

    manager.subscribe(session_id, _print_event)

    await manager.start(session)

    state = session.state
    print("\n" + "=" * 40)
    print(f"Rounds:   {state.current_round}")
    print(f"Messages: {state.message_count}")
    print(f"Cost:     ${state.total_cost_usd:.4f}")
    print(f"Status:   {state.status}")
    print("=" * 40)

    await dispose(engine)
    return 0


def _export(args: argparse.Namespace) -> int:
    import uuid as _uuid

    from thinktank.storage.db import make_engine, make_session_factory
    from thinktank.storage.repositories import EventStore

    try:
        session_id = _uuid.UUID(args.session)
    except ValueError:
        print(f"error: {args.session!r} is not a valid session id", file=sys.stderr)
        return 2

    async def _read() -> str:
        engine = make_engine()
        store = EventStore(make_session_factory(engine))
        try:
            return await store.export(session_id, format=args.format)
        finally:
            await engine.dispose()

    text = asyncio.run(_read())

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {len(text)} bytes to {args.out}")
    else:
        sys.stdout.write(text)
    return 0


async def _reembed(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = make_engine(load_vec=True)
    await init_db(engine)
    embedder = build_embedding_provider(
        backend=settings.embedding_backend,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
    )
    backend = SQLiteMemoryBackend(engine, embedder)
    try:
        count = await backend.reembed(embedder)
    finally:
        await dispose(engine)
    print(f"re-embedded {count} memory item(s) with {embedder.name} (dim={embedder.dim})")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    if getattr(args, "fake", False):
        os.environ["THINKTANK_FAKE_LLM"] = "1"
    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port
    print(f"ThinkTank hub on http://{host}:{port}  (Ctrl-C to stop)")
    uvicorn.run(
        "thinktank.web.app:asgi_factory",
        factory=True,
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="thinktank", description="Multi-party AI/human debate system.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a debate session from a YAML template.")
    run_p.add_argument("session", type=Path, help="Path to the session YAML template.")
    run_p.add_argument("--fake", action="store_true", help="Use the offline FakeLLM (no API calls).")
    run_p.add_argument(
        "--human-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for a human turn before skipping (headless: use a small value).",
    )
    run_p.add_argument("-q", "--quiet", action="store_true", help="Suppress live transcript printing.")

    serve_p = sub.add_parser("serve", help="Start the web hub (M2).")
    serve_p.add_argument("--host", default=None, help="Bind host (default from settings, 127.0.0.1).")
    serve_p.add_argument("--port", type=int, default=None, help="Bind port (default from settings, 8080).")
    serve_p.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (development).")
    serve_p.add_argument("--fake", action="store_true", help="Use the offline FakeLLM (no API calls).")
    export_p = sub.add_parser("export", help="Export a session's event stream (DESIGN.md §14.1, M6).")
    export_p.add_argument("session", help="Session id (UUID) to export.")
    export_p.add_argument("--format", choices=["jsonl", "csv"], default="jsonl", help="Output format (default jsonl).")
    export_p.add_argument("--out", default=None, help="Write to this file instead of stdout.")
    sub.add_parser(
        "reembed",
        help="Recompute memory embeddings after a model change (DESIGN.md §11, M3).",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if not getattr(args, "quiet", False) else logging.WARNING, stream=sys.stderr)

    if args.command == "run":
        return asyncio.run(_run(args))
    if args.command == "serve":
        return _serve(args)
    if args.command == "export":
        return _export(args)
    if args.command == "reembed":
        return asyncio.run(_reembed(args))
    print(f"`thinktank {args.command}` is available in a later milestone (see DESIGN.md §18).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
