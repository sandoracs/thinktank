"""Session manager: lifecycle, participant construction, crash recovery
(DESIGN.md §3, §9.2).

Owns the single :class:`EventBus` and :class:`EventStore` for the process and
builds a :class:`SessionEngine` per session. On startup it marks any session
left ``running`` as ``interrupted`` so a crash never leaves a zombie session
(DESIGN.md §9.2); replay/resume rebuilds state by projecting the stored events
through the same code path used live.
"""

from __future__ import annotations

import uuid
from typing import Any

from roundtable.core.bus import EventBus
from roundtable.core.context import ContextBuilder, ParticipantInfo
from roundtable.core.engine import SessionEngine
from roundtable.core.state import SessionState, project
from roundtable.domain.events import EventType, make_event
from roundtable.domain.models import AgentConfig, SessionConfig
from roundtable.llm.client import LLMClient
from roundtable.memory.base import MemoryBackend
from roundtable.participants.ai_agent import AIAgent
from roundtable.participants.base import Participant
from roundtable.participants.human import HumanParticipant
from roundtable.plugins.registry import build_strategy
from roundtable.storage.repositories import EventStore
from roundtable.strategies.base import TurnStrategy


class SessionManager:
    def __init__(
        self,
        *,
        store: EventStore,
        llm: LLMClient,
        memory: MemoryBackend | None = None,
        context_builder: ContextBuilder | None = None,
        default_model: str = "claude-3-5-sonnet-latest",
        human_timeout_s: float = 60.0,
    ) -> None:
        self._store = store
        self._bus = EventBus(append=store.append)
        self._llm = llm
        self._memory = memory
        self._context_builder = context_builder or ContextBuilder()
        self._default_model = default_model
        self._human_timeout_s = human_timeout_s
        self._engines: dict[uuid.UUID, SessionEngine] = {}

    @property
    def human_timeout_s(self) -> float:
        """Seconds a human has to answer before the turn is skipped (DESIGN.md §8.1)."""
        return self._human_timeout_s

    # -- lifecycle ---------------------------------------------------------
    async def recover_on_start(self) -> int:
        """Mark any leftover ``running`` sessions as ``interrupted``."""
        return await self._store.mark_interrupted_on_start()

    async def create_session(
        self,
        config: SessionConfig,
        agents: dict[str, AgentConfig],
    ) -> SessionEngine:
        session_id = uuid.uuid4()
        await self._store.create_session(session_id, config)
        created = await self._bus.emit(session_id, make_event(EventType.SESSION_CREATED, config=config))

        strategy: TurnStrategy = build_strategy(config.strategy, llm=self._llm)
        events = await self._store.get_events(session_id)
        state = project(session_id, events)

        engine = SessionEngine(
            session_id=session_id,
            config=config,
            strategy=strategy,
            participants={},
            state=state,
            bus=self._bus,
            store=self._store,
        )
        participants = self._build_participants(config, state, engine.emit, agents)
        engine.bind_participants(participants)
        await strategy.on_event(created)

        self._engines[session_id] = engine
        return engine

    def _build_participants(
        self,
        config: SessionConfig,
        state: SessionState,
        emit: Any,
        agents: dict[str, AgentConfig],
    ) -> dict[str, Participant]:
        infos: list[ParticipantInfo] = []
        for ref in config.participants:
            pid = ref.participant_id
            if ref.participant_kind == "ai":
                cfg = agents.get(pid)
                name = cfg.persona.name if cfg else pid
                role = cfg.persona.role if cfg else ""
            else:
                name, role = pid, ""
            infos.append(ParticipantInfo(id=pid, name=name, description=role))

        participants: dict[str, Participant] = {}
        for ref in config.participants:
            pid = ref.participant_id
            if ref.participant_kind == "ai":
                cfg = agents.get(pid)
                if cfg is None:
                    msg = f"Agent template {pid!r} referenced but not provided"
                    raise ValueError(msg)
                cfg = cfg.with_defaults(self._default_model)
                participants[pid] = AIAgent(
                    config=cfg,
                    session_config=config,
                    state=state,
                    participants=infos,
                    llm=self._llm,
                    context_builder=self._context_builder,
                    memory=self._memory,
                    emit=emit,
                )
            elif ref.participant_kind == "human":
                participants[pid] = HumanParticipant(
                    pid,
                    human_timeout_s=self._human_timeout_s,
                    is_moderator=(config.moderator == pid),
                )
            else:
                msg = f"RemoteAgent participant {pid!r} is not available until M5"
                raise NotImplementedError(msg)
        return participants

    # -- control -----------------------------------------------------------
    def engine_for(self, session_id: uuid.UUID) -> SessionEngine:
        engine = self._engines.get(session_id)
        if engine is None:
            msg = f"No active engine for session {session_id}"
            raise KeyError(msg)
        return engine

    async def start(self, engine: SessionEngine) -> None:
        await engine.run()

    def pause(self, engine: SessionEngine) -> None:
        engine.pause()

    def resume(self, engine: SessionEngine) -> None:
        engine.resume()

    def stop(self, engine: SessionEngine, reason: str = "manual") -> None:
        engine.stop(reason)

    # -- approvals (DESIGN.md §12.3, M6) ----------------------------------
    async def list_approvals(
        self, session_id: uuid.UUID, agent_id: str | None = None
    ) -> list[dict[str, object]]:
        return await self._store.list_approvals(session_id, agent_id=agent_id)

    async def decide_approval(self, session_id: uuid.UUID, agent_id: str, decision: str) -> bool:
        return await self.engine_for(session_id).decide_approval(agent_id, decision)

    # -- inspection --------------------------------------------------------
    async def get_state(self, session_id: uuid.UUID) -> SessionState:
        events = await self._store.get_events(session_id)
        return project(session_id, events)

    def subscribe(self, session_id: uuid.UUID, subscriber: Any) -> Any:
        return self._bus.subscribe(session_id, subscriber)
