"""Tests for the real Clip Scout + Critic nodes with a mocked LLM.

Covers the agentic core: the reject->revise->approve loop, round-counter
increments, termination at MAX_CRITIC_ROUNDS, and code-level timestamp
validation catching an out-of-bounds clip.

All LLM calls go through the single seam `app.agents.llm.structured_invoke`,
which we patch with a schema dispatcher so the whole graph runs offline.
"""

from __future__ import annotations

from typing import Callable

import pytest

from app.agents import llm as llm_mod
from app.agents.graph import build_graph
from app.agents.state import (
    ClipPick,
    ClipPickList,
    ContentState,
    CriticNote,
    CriticReview,
    HooksOutput,
    SEOOutput,
    TitleSmithOutput,
)
from app.ingestion.chunker import build_content_map
from app.ingestion.transcript import Segment, Transcript

USAGE = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "cost_usd": 0.0001}


def _state(duration_s: float = 600.0) -> ContentState:
    segs = [
        Segment(start_s=i * 10, end_s=(i + 1) * 10, text=f"Sentence number {i} is complete.")
        for i in range(int(duration_s // 10))
    ]
    t = Transcript(video_id="v", language="en", segments=segs, total_duration_s=duration_s)
    return {
        "transcript": t,
        "content_map": build_content_map(t),
        "content_type": "unknown",
        "node_trace": [],
        "errors": [],
        "token_usage": {},
    }


def _two_picks() -> ClipPickList:
    return ClipPickList(clips=[
        ClipPick(start_s=0, end_s=30, transcript_text="A self contained thought that ends.",
                 reason_chosen="strong open", hook_potential=8, completeness=8, shareability=8),
        ClipPick(start_s=40, end_s=70, transcript_text="And then the thing that",
                 reason_chosen="cut off", hook_potential=6, completeness=4, shareability=5),
    ])


def _dispatcher(critic_fn: Callable[[], CriticReview]) -> Callable:
    """Build a fake structured_invoke that returns canned output per schema."""

    def _fake(schema, messages, **kwargs):
        if schema is ClipPickList:
            return _two_picks(), USAGE
        if schema is CriticReview:
            return critic_fn(), USAGE
        if schema is HooksOutput:
            return HooksOutput(clip_hooks=[]), USAGE
        if schema is TitleSmithOutput:
            return TitleSmithOutput(main_video_titles=["A title"], per_clip=[]), USAGE
        if schema is SEOOutput:
            return SEOOutput(description_md="desc", tags=["a"], hashtags=["#a"],
                             shorts_caption="s", reels_caption="r"), USAGE
        raise AssertionError(f"unexpected schema {schema}")

    return _fake


# --------------------------------------------------------------------------- #
# 1) reject -> revise -> approve, loop runs once, round counter increments
# --------------------------------------------------------------------------- #
def test_loop_reject_then_approve(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def critic_fn() -> CriticReview:
        calls["n"] += 1
        if calls["n"] == 1:
            return CriticReview(notes=[
                CriticNote(clip_id="clip_1", verdict="approve", reasons=["clean"]),
                CriticNote(clip_id="clip_2", verdict="reject", reasons=["mid-thought cutoff"]),
            ])
        return CriticReview(notes=[
            CriticNote(clip_id="clip_1", verdict="approve", reasons=["clean"]),
            CriticNote(clip_id="clip_2", verdict="approve", reasons=["fixed"]),
        ])

    monkeypatch.setattr(llm_mod, "structured_invoke", _dispatcher(critic_fn))
    final = build_graph(max_critic_rounds=2).invoke(_state())
    trace = final["node_trace"]

    assert trace.count("clip_scout") == 2
    assert trace.count("critic") == 2
    assert final["critic_round"] == 2
    assert "clip_scout.r0" in final["token_usage"]
    assert "critic.r1" in final["token_usage"] and "critic.r2" in final["token_usage"]
    assert len(final["approved_clips"]) == 2


# --------------------------------------------------------------------------- #
# 2) critic never approves -> loop terminates at MAX_CRITIC_ROUNDS
# --------------------------------------------------------------------------- #
def test_loop_terminates_at_max(monkeypatch: pytest.MonkeyPatch) -> None:
    def critic_fn() -> CriticReview:
        return CriticReview(notes=[
            CriticNote(clip_id="clip_1", verdict="approve", reasons=["clean"]),
            CriticNote(clip_id="clip_2", verdict="reject", reasons=["weak"]),
        ])

    monkeypatch.setattr(llm_mod, "structured_invoke", _dispatcher(critic_fn))
    final = build_graph(max_critic_rounds=2).invoke(_state())
    trace = final["node_trace"]

    assert trace.count("critic") == 2
    assert trace.count("clip_scout") == 2
    assert final["critic_round"] == 2
    assert [a.candidate.clip_id for a in final["approved_clips"]] == ["clip_1"]
    assert "assembler" in trace


# --------------------------------------------------------------------------- #
# 3) everything rejected -> top-2 fallback keeps the pipeline alive
# --------------------------------------------------------------------------- #
def test_all_rejected_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def critic_fn() -> CriticReview:
        return CriticReview(notes=[
            CriticNote(clip_id="clip_1", verdict="reject", reasons=["weak"]),
            CriticNote(clip_id="clip_2", verdict="reject", reasons=["weak"]),
        ])

    monkeypatch.setattr(llm_mod, "structured_invoke", _dispatcher(critic_fn))
    final = build_graph(max_critic_rounds=2).invoke(_state())
    assert len(final["approved_clips"]) == 2
    assert any("fell back" in e for e in final["errors"])


# --------------------------------------------------------------------------- #
# 4) timestamp validation clamps out-of-bounds and out-of-duration clips
# --------------------------------------------------------------------------- #
def test_timestamp_validation_clamps(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agents import clip_scout as cs_mod

    picks = ClipPickList(clips=[
        ClipPick(start_s=10, end_s=9999, transcript_text="way too long and out of bounds.",
                 reason_chosen="x", hook_potential=7, completeness=7, shareability=7),
        ClipPick(start_s=100, end_s=105, transcript_text="too short.",
                 reason_chosen="y", hook_potential=6, completeness=6, shareability=6),
    ])
    monkeypatch.setattr(llm_mod, "structured_invoke", lambda schema, msgs, **kw: (picks, USAGE))

    out = cs_mod.clip_scout_node(_state(600))
    cands = out["clip_candidates"]

    assert cands[0].start_s == 10.0
    assert cands[0].end_s == 85.0  # 10 + 75s max
    assert cands[0].end_s <= 600
    assert cands[1].duration_s >= 15.0
    assert cands[1].end_s <= 600


# --------------------------------------------------------------------------- #
# 5) long video is chunked (map-reduce) — clip_scout makes several LLM calls
# --------------------------------------------------------------------------- #
def test_long_video_is_chunked(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agents import clip_scout as cs_mod

    calls = {"n": 0}

    def fake(schema, msgs, **kw):
        calls["n"] += 1
        # One unique-scored pick per chunk so we can see the reduce keep the best.
        return ClipPickList(clips=[
            ClipPick(start_s=calls["n"] * 60, end_s=calls["n"] * 60 + 30,
                     transcript_text=f"clip from chunk {calls['n']}",
                     reason_chosen="x", hook_potential=calls["n"] % 10 + 1,
                     completeness=5, shareability=5),
        ]), USAGE

    monkeypatch.setattr(llm_mod, "structured_invoke", fake)

    # ~6000 words (~8000 tokens) -> above CHUNK_THRESHOLD -> multiple chunks.
    segs = [Segment(start_s=i * 4, end_s=(i + 1) * 4, text="word " * 20) for i in range(300)]
    t = Transcript(video_id="long", language="en", segments=segs, total_duration_s=1200)
    state: ContentState = {
        "transcript": t, "content_map": build_content_map(t), "content_type": "unknown",
        "node_trace": [], "errors": [], "token_usage": {},
    }

    out = cs_mod.clip_scout_node(state)
    assert calls["n"] >= 2  # transcript was split into multiple chunks
    assert out["clip_candidates"]  # candidates were merged from the chunks
    # Usage was accumulated across all chunk calls.
    assert out["token_usage"]["clip_scout.r0"]["total_tokens"] >= USAGE["total_tokens"] * 2
