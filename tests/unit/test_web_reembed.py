"""Tests for the embedding model-mismatch helper (DESIGN.md §11, M3).

The hub signals an error at startup when stored memory vectors were produced by
a different model than the one now configured; ``roundtable reembed`` is the
remedy. The pure helper decides when to signal.
"""

from __future__ import annotations

from roundtable.web.app import embedding_mismatch_message


def test_no_mismatch_when_matching() -> None:
    assert embedding_mismatch_message("fake", "fake") is None


def test_no_mismatch_when_nothing_stored() -> None:
    assert embedding_mismatch_message(None, "litellm:paraphrase-multilingual-MiniLM-L12-v2") is None


def test_mismatch_when_model_changed() -> None:
    msg = embedding_mismatch_message("fake", "litellm:paraphrase-multilingual-MiniLM-L12-v2")
    assert msg is not None
    assert "reembed" in msg
    assert "fake" in msg
