"""Reciprocal Rank Fusion unit tests (DESIGN.md §17: RRF-fúzió)."""

from __future__ import annotations

from roundtable.memory.base import RRF_K, reciprocal_rank_fusion


def test_single_ranking_preserves_order() -> None:
    scores = reciprocal_rank_fusion([1, 2, 3])
    assert scores[1] > scores[2] > scores[3]

def test_ids_present_in_both_rankings_win() -> None:
    # 7 is rank 0 in the first list and rank 1 in the second; 1 is rank 1
    # (first list only) and 2 is rank 2 (first list) + rank 1 (second list).
    scores = reciprocal_rank_fusion([7, 1, 2, 3], [7, 2])
    assert scores[7] > scores[2]
    assert scores[2] > scores[1]


def test_absent_ids_contribute_nothing() -> None:
    scores = reciprocal_rank_fusion([1, 2], [])
    assert set(scores) == {1, 2}
    assert reciprocal_rank_fusion([]) == {}


def test_k_constant_is_60() -> None:
    scores = reciprocal_rank_fusion([1])
    assert scores[1] == 1.0 / RRF_K
