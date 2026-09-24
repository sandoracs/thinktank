"""Drift policies (DESIGN.md §8.4, §12.3).

A policy maps a ``ReflectionResult`` proposal + a ``DriftConfig`` to a
:class:`PolicyDecision`. The engine then either applies, clamps, rejects, or
routes the change for approval — and records whichever as an event so the
inspector's drift timeline is exact (DESIGN.md §12.3).

The four built-in modes are always importable here; external policies can be
registered via the ``roundtable.drift_policies`` entry point (M5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from roundtable.domain.models import DriftConfig, PersonaState, Stance
from roundtable.persona.reflection import ReflectionResult, apply_updates

Action = Literal["apply", "reject", "clamp", "needs_approval"]


def _clamp_value(value: float, old: float, max_delta: float) -> float:
    """Limit movement away from ``old`` to ``max_delta``."""
    delta = value - old
    if delta > max_delta:
        return old + max_delta
    if delta < -max_delta:
        return old - max_delta
    return value


@dataclass
class PolicyDecision:
    action: Action
    new_state: PersonaState | None = None
    reason: str = ""
    notes: list[str] = field(default_factory=list)


class DriftPolicy(ABC):
    name: ClassVar[str]

    @abstractmethod
    def evaluate(
        self,
        current: PersonaState,
        proposal: ReflectionResult,
        config: DriftConfig,
    ) -> PolicyDecision:
        """Decide how to treat the proposal. Pure: no events, no side effects."""
        ...


class LockedPolicy(DriftPolicy):
    name = "locked"

    def evaluate(self, current: PersonaState, proposal: ReflectionResult, config: DriftConfig) -> PolicyDecision:
        return PolicyDecision(action="reject", reason="locked", new_state=current.clone())


class FreePolicy(DriftPolicy):
    name = "free"

    def evaluate(self, current: PersonaState, proposal: ReflectionResult, config: DriftConfig) -> PolicyDecision:
        return PolicyDecision(action="apply", new_state=apply_updates(current, proposal))


class BoundedPolicy(DriftPolicy):
    name = "bounded"

    def evaluate(self, current: PersonaState, proposal: ReflectionResult, config: DriftConfig) -> PolicyDecision:
        new = current.clone()
        notes: list[str] = []
        changes = 0
        for update in proposal.stance_updates:
            if changes >= config.max_stance_changes_per_round:
                notes.append(
                    f"skipped stance change for {update.question_id} "
                    f"(limit {config.max_stance_changes_per_round})"
                )
                continue
            old = current.stances.get(update.question_id)
            if old is not None:
                clamped = _clamp_value(update.new_confidence, old.confidence, config.max_confidence_delta)
                if clamped != update.new_confidence:
                    notes.append(
                        f"clamped confidence {update.question_id}: "
                        f"{old.confidence:.2f} -> {clamped:.2f} (asked {update.new_confidence:.2f})"
                    )
                new.stances[update.question_id] = Stance(position=update.new_position, confidence=clamped)
            else:
                new.stances[update.question_id] = Stance(position=update.new_position, confidence=update.new_confidence)
            changes += 1
        for update in proposal.attitude_updates:
            old = current.attitudes.get(update.participant_id, 0.0)
            clamped = _clamp_value(update.new_value, old, config.max_attitude_delta)
            if clamped != update.new_value:
                notes.append(
                    f"clamped attitude {update.participant_id}: "
                    f"{old:+.2f} -> {clamped:+.2f} (asked {update.new_value:+.2f})"
                )
            new.attitudes[update.participant_id] = clamped
        if proposal.mood is not None:
            new.mood = proposal.mood
        action: Action = "clamp" if notes else "apply"
        return PolicyDecision(action=action, new_state=new, notes=notes)


class ApprovedPolicy(DriftPolicy):
    name = "approved"

    def evaluate(self, current: PersonaState, proposal: ReflectionResult, config: DriftConfig) -> PolicyDecision:
        return PolicyDecision(
            action="needs_approval",
            new_state=apply_updates(current, proposal),
            reason="awaiting approval",
        )


_BUILTIN: dict[str, type[DriftPolicy]] = {
    LockedPolicy.name: LockedPolicy,
    FreePolicy.name: FreePolicy,
    BoundedPolicy.name: BoundedPolicy,
    ApprovedPolicy.name: ApprovedPolicy,
}


def build_policy(mode: str) -> DriftPolicy:
    """Return a policy instance for a mode name (built-ins only in M0-M4)."""
    cls = _BUILTIN.get(mode)
    if cls is None:
        available = ", ".join(sorted(_BUILTIN))
        msg = f"Unknown drift policy {mode!r}; available: {available}"
        raise ValueError(msg)
    return cls()
