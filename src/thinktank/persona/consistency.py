"""Persona-consistency judge (M6).

Opt-in per agent: a judge model scores the candidate speech against the frozen
persona core on a 1-5 scale and explains why. Below the agent's threshold the
speech is regenerated once with the judge's feedback. Either way a
``ConsistencyViolation`` event is recorded. Off by default because it is a
costly extra model call.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from thinktank.domain.models import PersonaCore
from thinktank.llm.client import ChatMessage

JUDGE_SYSTEM = (
    "You are an impartial consistency judge for an AI debate participant. "
    "Given the participant's frozen persona core and a candidate utterance they "
    "are about to post, judge how well the utterance stays faithful to that "
    "persona. Score from 1 (clearly contradicts the core) to 5 (fully "
    "consistent). Be strict about stated values and boundaries. Reply with the "
    "score and a one-sentence justification."
)


class ConsistencyJudgment(BaseModel):
    """Structured output of the consistency judge."""

    score: int = Field(ge=1, le=5)
    justification: str = ""


def consistency_messages(*, core: PersonaCore, candidate: str, feedback: str = "") -> list[ChatMessage]:
    """Build the judge prompt (or, with ``feedback``, the regeneration prompt)."""
    user = f"Persona core:\n{core.render()}\n\nCandidate utterance:\n{candidate}"
    if feedback:
        rewrite_hint = (
            "Rewrite the utterance so it fully respects the persona core, keeping the same substance and length."
        )
        user += f"\n\nA judge found it inconsistent: {feedback}\n{rewrite_hint}"
        system = "You are an AI debate participant staying true to your persona. Rewrite only as instructed."
        return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]
    return [
        ChatMessage(role="system", content=JUDGE_SYSTEM),
        ChatMessage(role="user", content=user),
    ]
