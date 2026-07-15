"""Unit tests for the eval metric helpers (no LLM)."""

from __future__ import annotations

from evals.run_evals import (
    clip_covers_golden,
    golden_recall,
    intersection_s,
    selection_precision,
)


def test_intersection() -> None:
    assert intersection_s(0, 10, 5, 15) == 5
    assert intersection_s(0, 10, 20, 30) == 0
    assert intersection_s(10, 60, 0, 100) == 50


def test_clip_covers_golden_threshold() -> None:
    golden = {"start_s": 100, "end_s": 160}  # 60s
    # 40s overlap = 67% -> hit
    assert clip_covers_golden({"start_s": 120, "end_s": 200}, golden) is True
    # 20s overlap = 33% -> miss
    assert clip_covers_golden({"start_s": 140, "end_s": 165}, golden) is False
    # exactly 50%
    assert clip_covers_golden({"start_s": 100, "end_s": 130}, golden) is True


def test_golden_recall() -> None:
    golden = [
        {"start_s": 0, "end_s": 30},
        {"start_s": 100, "end_s": 130},
        {"start_s": 200, "end_s": 230},
    ]
    clips = [
        {"start_s": 0, "end_s": 30},     # covers #1
        {"start_s": 100, "end_s": 130},  # covers #2
    ]
    res = golden_recall(clips, golden)
    assert res == {"hits": 2, "total": 3, "recall": 2 / 3}


def test_golden_recall_empty() -> None:
    assert golden_recall([], [])["recall"] == 0.0


def test_selection_precision() -> None:
    golden = [{"start_s": 0, "end_s": 30}, {"start_s": 100, "end_s": 130}]
    clips = [
        {"start_s": 0, "end_s": 30},      # hits golden #1
        {"start_s": 500, "end_s": 530},   # hits nothing
    ]
    assert selection_precision(clips, golden) == 0.5
    assert selection_precision([], golden) == 0.0
