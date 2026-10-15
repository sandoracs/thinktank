"""Session language mapping."""

from __future__ import annotations

from thinktank.core.language import language_instruction, language_name
from thinktank.memory.distill import episodic_messages, long_term_messages


def test_language_name_maps_codes_and_passes_through() -> None:
    assert language_name("hu") == "Hungarian"
    assert language_name("en-GB") == "English"
    assert language_name("Esperanto") == "Esperanto"
    assert language_name("") == ""
    assert language_name(None) == ""


def test_distill_prompts_carry_language() -> None:
    assert language_instruction("hu") == "Write your answer in Hungarian."
    ep = episodic_messages(agent_name="A", topic="t", transcript="x", language="hu")
    lt = long_term_messages(agent_name="A", topic="t", episodic_summary="x", language="hu")
    assert "Hungarian" in ep[0].content
    assert "Hungarian" in lt[0].content
    assert "Hungarian" not in episodic_messages(agent_name="A", topic="t", transcript="x")[0].content
