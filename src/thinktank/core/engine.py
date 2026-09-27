"""The debate engine.

Drives one session: moderator opens, then rounds of turns until a stop
condition holds, then the moderator closes. Every fact is emitted through
:meth:`emit`, which (1) persists it with an assigned ``seq``, (2) applies it to
the live :class:`SessionState`, and (3) feeds it to the strategy — so the live
path and the replay path (which rebuilds state from the same events) stay
identical.

Error handling: a failed ``speak`` becomes ``Error`` + ``TurnSkipped(error)``
and the debate continues; three consecutive failures disable the participant
for the rest of the session.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from thinktank.core.bus import EventBus
from thinktank.core.state import SessionState, apply_event
from thinktank.domain.events import Event, EventType, make_event
from thinktank.domain.models import DriftMode, Message, PersonaState, SessionConfig
from thinktank.participants.base import Participant, TurnContext
from thinktank.persona.manager import PersonaManager
from thinktank.persona.reflection import ReflectionResult, apply_updates
from thinktank.storage.repositories import EventStore
from thinktank.strategies.base import TurnStrategy

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 3


class PauseGate:
    """Blocks the main loop while paused; ``stop`` unblocks it to let teardown run."""

    def __init__(self) -> None:
        self._running = asyncio.Event()
        self._running.set()
        self._stopped = False

    def pause(self) -> None:
        self._running.clear()

    def resume(self) -> None:
        self._running.set()

    def stop(self) -> None:
        self._stopped = True
        self._running.set()

    @property
    def is_running(self) -> bool:
        return self._running.is_set() and not self._stopped

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    async def wait(self) -> None:
        await self._running.wait()


class SessionEngine:
    def __init__(
        self,
        *,
        session_id: uuid.UUID,
        config: SessionConfig,
        strategy: TurnStrategy,
        participants: dict[str, Participant],
        state: SessionState,
        bus: EventBus,
        store: EventStore,
        persona_manager: PersonaManager | None = None,
    ) -> None:
        self._session_id = session_id
        self._config = config
        self._strategy = strategy
        self._participants = participants
        self._state = state
        self._bus = bus
        self._store = store
        self._gate = PauseGate()
        self._stop_reason = ""
        self._started_at: float | None = None
        self._consecutive_errors: dict[str, int] = {}
        self._persona_manager = persona_manager or PersonaManager()

    def bind_participants(self, participants: dict[str, Participant]) -> None:
        """Attach participant instances post-construction (breaks the build cycle)."""
        self._participants = participants

    # -- control -----------------------------------------------------------
    def pause(self) -> None:
        self._gate.pause()

    def resume(self) -> None:
        self._gate.resume()

    def stop(self, reason: str = "manual") -> None:
        self._stop_reason = reason
        self._gate.stop()

    async def raise_hand(self, participant_id: str) -> Event:
        """Record a raised hand (called by the hub when a human taps the button)."""
        return await self.emit(EventType.HAND_RAISED, participant_id=participant_id)


    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def participants(self) -> dict[str, Participant]:
        return self._participants

    @property
    def session_id(self) -> uuid.UUID:
        return self._session_id

    # -- emit --------------------------------------------------------------
    async def emit(self, etype: EventType, **payload: object) -> Event:
        event = make_event(etype, **payload)
        stored = await self._bus.emit(self._session_id, event)
        apply_event(self._state, stored)
        await self._strategy.on_event(stored)
        return stored

    # -- main loop ---------------------------------------------------------
    async def run(self) -> None:
        await self.emit(EventType.SESSION_STARTED)
        self._started_at = time.monotonic()
        await self._moderator_open()
        try:
            round_no = 1
            while True:
                if (await self._should_stop())[0]:
                    break
                max_rounds = self._config.stop.max_rounds
                if max_rounds is not None and round_no > max_rounds:
                    # Cap reached: stop before starting the next round so the
                    # event log never contains a RoundStarted without a RoundEnded.
                    self.stop("max_rounds")
                    break
                await self._gate.wait()
                await self._start_round(round_no)
                await self._play_round()
                if self._state.round_complete:
                    await self._end_round()
                round_no += 1
        finally:
            await self._moderator_close()
            await self._finish()

    async def _play_round(self) -> None:
        while not self._state.round_complete and not (await self._should_stop())[0]:
            await self._gate.wait()
            speaker_id = await self._strategy.next_speaker(self._state)
            if speaker_id is None:
                self._state.round_complete = True
                break
            await self.emit(
                EventType.TURN_ASSIGNED, speaker_id=speaker_id, strategy=self._strategy.name
            )
            ctx = TurnContext(session_id=self._session_id, round=self._state.current_round)
            await self._turn(speaker_id, ctx)

    async def _turn(self, speaker_id: str, ctx: TurnContext) -> None:
        participant = self._participants.get(speaker_id)
        if participant is None:
            # Stale speaker id (e.g. a raised hand for a participant no longer
            # in the session): skip rather than crash the run loop, and clear
            # the hand so it does not wedge every subsequent round forever.
            await self.emit(EventType.TURN_SKIPPED, speaker_id=speaker_id, reason="unknown_participant")
            await self._maybe_lower_hand(speaker_id)
            return
        try:
            message = await participant.speak(ctx)
        except Exception as exc:
            logger.exception("speak failed for %s", speaker_id)
            await self.emit(EventType.ERROR, component="speak", message=str(exc))
            await self.emit(EventType.TURN_SKIPPED, speaker_id=speaker_id, reason="error")
            await self._register_error(speaker_id)
            await self._maybe_lower_hand(speaker_id)
            return
        if message is None:
            reason = "human_timeout" if participant.kind == "human" else "passed"
            await self.emit(EventType.TURN_SKIPPED, speaker_id=speaker_id, reason=reason)
            await self._maybe_lower_hand(speaker_id)
            return
        self._consecutive_errors[speaker_id] = 0
        await self._post(message)

    async def _maybe_lower_hand(self, participant_id: str) -> None:
        if participant_id in self._state.hands_raised:
            await self.emit(EventType.HAND_LOWERED, participant_id=participant_id)

    async def _post(self, message: Message) -> None:
        await self.emit(EventType.MESSAGE_POSTED, message=message)
        results = await asyncio.gather(
            *(p.observe(message) for p in self._participants.values()),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logger.warning("observe failed: %s", result)
        await self._maybe_lower_hand(message.speaker_id)

    async def _end_round(self) -> None:
        round_no = self._state.current_round
        await self.emit(EventType.ROUND_ENDED, round=round_no)
        await self._reflect_round(round_no)
        self._state.round_complete = False

    # -- reflection (M4) ---------------------------------
    async def _reflect_round(self, round_no: int) -> None:
        """Ask each AI participant to reflect; evaluate + record the proposal."""
        if round_no % max(1, self._config.reflection_every_rounds) != 0:
            return
        for pid, participant in self._participants.items():
            if participant.kind != "ai" or pid in self._state.disabled:
                continue
            drift = getattr(participant, "drift", None)
            if (
                drift is not None
                and drift.mode is DriftMode.LOCKED
                and not drift.shadow_reflection
            ):
                # LOCKED without shadow: the design does not even run reflection
                # — nothing to propose, nothing to log.
                continue
            try:
                proposal = await participant.reflect(self._state)
            except Exception as exc:
                logger.exception("reflect failed for %s", pid)
                await self.emit(EventType.ERROR, component="reflect", message=str(exc))
                continue
            if proposal is None:
                continue
            # Shadow log: every non-empty proposal is recorded, whether or not
            # the drift policy later applies it (shadow mode).
            shadow = drift is not None and bool(drift.shadow_reflection)
            proposed = await self.emit(
                EventType.REFLECTION_PROPOSED,
                agent_id=pid,
                proposal=proposal.model_dump(),
                round=round_no,
                shadow=shadow,
            )
            await self._settle_reflection(pid, proposal, proposed.seq)

    def _current_persona(self, pid: str) -> PersonaState:
        if pid in self._state.persona_states:
            return self._state.persona_states[pid]
        participant = self._participants.get(pid)
        initial = getattr(participant, "initial_state", None)
        return initial if isinstance(initial, PersonaState) else PersonaState()

    async def _settle_reflection(
        self, pid: str, proposal: ReflectionResult, cause_seq: int
    ) -> None:
        drift = getattr(self._participants.get(pid), "drift", None)
        if drift is None:
            return
        decision = self._persona_manager.evaluate(self._current_persona(pid), proposal, drift)
        if decision.action in ("apply", "clamp") and decision.new_state is not None:
            etype = (
                EventType.PERSONA_UPDATED
                if decision.action == "apply"
                else EventType.PERSONA_UPDATE_CLAMPED
            )
            await self._commit_persona(pid, decision.new_state, etype, cause_seq, decision.notes)
        elif decision.action == "needs_approval":
            # One outstanding request per agent: a later reflection while the
            # first is still pending must not silently pile up a second row
            # that decide_approval would never look at.
            existing = await self._store.list_approvals(self._session_id, agent_id=pid)
            if any(a["status"] == "pending" for a in existing):
                return
            await self.emit(
                EventType.APPROVAL_REQUESTED,
                agent_id=pid,
                proposal=proposal.model_dump(),
                cause_seq=cause_seq,
            )
        else:
            await self.emit(
                EventType.PERSONA_UPDATE_REJECTED,
                agent_id=pid,
                reason=decision.reason or "rejected",
                cause_seq=cause_seq,
                notes=decision.notes,
            )

    async def _commit_persona(
        self,
        pid: str,
        new_state: PersonaState,
        etype: EventType,
        cause_seq: int,
        notes: list[str] | None = None,
    ) -> None:
        version = await self._store.next_persona_version(pid, self._session_id)
        await self._store.append_persona_version(
            pid, self._session_id, version, new_state.model_dump(), cause_seq
        )
        await self.emit(
            etype,
            agent_id=pid,
            version=version,
            state=new_state,
            cause_seq=cause_seq,
            notes=notes or [],
        )

    # -- approvals (M6) ---------------------------------
    async def decide_approval(self, agent_id: str, decision: str) -> bool:
        """Approve or reject a pending persona change (APPROVED drift mode).

        ``approve`` applies the stored proposal to the current persona state and
        records ``PersonaUpdated``; ``reject`` records ``PersonaUpdateRejected``.
        Either way an ``ApprovalDecided`` event is emitted and the pending row is
        closed. Returns ``True`` if a pending approval was found and decided.
        """
        approvals = await self._store.list_approvals(self._session_id, agent_id=agent_id)
        pending = next((a for a in approvals if a["status"] == "pending"), None)
        if pending is None:
            return False
        proposal = ReflectionResult.model_validate(pending["proposal"])
        current = self._current_persona(agent_id)
        if decision == "approve":
            new_state = apply_updates(current, proposal)
            version = await self._store.next_persona_version(agent_id, self._session_id)
            await self._store.append_persona_version(
                agent_id, self._session_id, version, new_state.model_dump(), None
            )
            await self.emit(
                EventType.PERSONA_UPDATED,
                agent_id=agent_id,
                version=version,
                state=new_state,
                notes=["approved"],
            )
        else:
            await self.emit(
                EventType.PERSONA_UPDATE_REJECTED,
                agent_id=agent_id,
                reason="not_approved",
                notes=["rejected by approver"],
            )
        await self._store.decide_approval_row(
            int(pending["id"]), "approved" if decision == "approve" else "rejected"  # pyright: ignore[reportArgumentType]
        )
        await self.emit(EventType.APPROVAL_DECIDED, agent_id=agent_id, decision=decision)
        return True

    async def _start_round(self, round_no: int) -> None:
        await self.emit(EventType.ROUND_STARTED, round=round_no)

    # -- moderator ---------------------------------------------------------
    def _moderator(self) -> Participant | None:
        if self._config.moderator is None:
            return None
        return self._participants.get(self._config.moderator)

    async def _moderator_open(self) -> None:
        moderator = self._moderator()
        if moderator is None or moderator.kind != "ai":
            return
        ctx = TurnContext(
            session_id=self._session_id,
            round=0,
            turn_instruction="Open the discussion: restate the topic and invite the first contributions.",
        )
        try:
            message = await moderator.speak(ctx)
        except Exception as exc:
            logger.exception("moderator open failed")
            await self.emit(EventType.ERROR, component="moderator_open", message=str(exc))
            return
        if message is not None:
            await self._post(message.model_copy(update={"kind": "moderator"}))

    async def _moderator_close(self) -> None:
        moderator = self._moderator()
        if moderator is None or moderator.kind != "ai":
            return
        ctx = TurnContext(
            session_id=self._session_id,
            round=self._state.current_round,
            turn_instruction="Close the discussion: summarise the main positions and any remaining disagreements.",
        )
        try:
            message = await moderator.speak(ctx)
        except Exception as exc:
            logger.exception("moderator close failed")
            await self.emit(EventType.ERROR, component="moderator_close", message=str(exc))
            return
        if message is not None:
            await self._post(message.model_copy(update={"kind": "moderator"}))

    # -- stop conditions ---------------------------------------------------
    async def _should_stop(self) -> tuple[bool, str]:
        if self._gate.is_stopped:
            return True, self._stop_reason or "manual"
        ids = self._config.participant_ids()
        if ids and all(pid in self._state.disabled for pid in ids):
            return True, "all_disabled"
        cfg = self._config.stop
        if cfg.max_messages is not None and self._state.message_count >= cfg.max_messages:
            return True, "max_messages"
        if cfg.max_cost_usd is not None and self._state.total_cost_usd >= cfg.max_cost_usd:
            return True, "cost_limit"
        if (
            cfg.max_duration_s is not None
            and self._started_at is not None
            and (time.monotonic() - self._started_at) >= cfg.max_duration_s
        ):
            return True, "max_duration"
        return False, ""

    async def _finish(self) -> None:
        reason = (await self._should_stop())[1] or "manual"
        for participant in self._participants.values():
            try:
                await participant.on_session_end()
            except Exception:
                logger.exception("on_session_end failed for %s", participant.id)
        await self.emit(EventType.SESSION_ENDED, reason=reason)

    # -- error handling ----------------------------------------------------
    async def _register_error(self, speaker_id: str) -> None:
        self._consecutive_errors[speaker_id] = self._consecutive_errors.get(speaker_id, 0) + 1
        if self._consecutive_errors[speaker_id] >= MAX_CONSECUTIVE_ERRORS:
            await self.emit(
                EventType.PARTICIPANT_DISABLED, participant_id=speaker_id, reason="repeated_errors"
            )
