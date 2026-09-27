"""Persona manager (DESIGN.md §12.3).

Resolves a drift policy from an agent's :class:`DriftConfig`, evaluates a
reflection proposal, and hands the :class:`PolicyDecision` back to the engine.
The engine is responsible for emitting the matching events
(``PersonaUpdated`` / ``PersonaUpdateClamped`` / ``PersonaUpdateRejected`` /
``ApprovalRequested``) and writing the new ``persona_versions`` row, keeping the
manager pure and unit-testable.
"""

from __future__ import annotations

from thinktank.domain.models import DriftConfig, PersonaState
from thinktank.persona.policies import DriftPolicy, PolicyDecision, build_policy
from thinktank.persona.reflection import ReflectionResult


class PersonaManager:
    def __init__(self, policy: DriftPolicy | None = None) -> None:
        self._policy = policy

    def policy_for(self, config: DriftConfig) -> DriftPolicy:
        if self._policy is not None:
            return self._policy
        return build_policy(config.mode.value)

    def evaluate(
        self,
        current: PersonaState,
        proposal: ReflectionResult,
        config: DriftConfig,
    ) -> PolicyDecision:
        if proposal.is_empty:
            return PolicyDecision(action="reject", reason="no_change")
        return self.policy_for(config).evaluate(current, proposal, config)
