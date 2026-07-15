"""Tests for guardrails and the generation agents (Hook/Title/SEO) with a
mocked LLM. Focus: length/count enforcement and the single corrective retry.
"""

from __future__ import annotations

import pytest

from app.agents import llm as llm_mod
from app.agents import hook_writer as hw_mod
from app.agents import seo_packager as seo_mod
from app.agents import title_smith as ts_mod
from app.agents.state import (
    ApprovedClip,
    ClipCandidate,
    ClipScores,
    ClipTitleSetLLM,
    ContentState,
    Hook,
    HookSet,
    HooksOutput,
    SEOOutput,
    TitleSmithOutput,
)
from app.guardrails.guards import (
    MAX_TITLE_CHARS,
    enforce_hashtags,
    enforce_tags,
    enforce_thumbnail_text,
    hook_violations,
    trim_hook,
    trim_title,
)
from app.ingestion.chunker import build_content_map
from app.ingestion.transcript import Segment, Transcript

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "cost_usd": 0.0}


# --------------------------------------------------------------------------- #
# Guardrail unit tests
# --------------------------------------------------------------------------- #
def test_trim_title_caps_at_60() -> None:
    long = "This is an extremely long YouTube title that definitely exceeds sixty characters for sure"
    out = trim_title(long)
    assert len(out) <= MAX_TITLE_CHARS
    assert not out.endswith(" ")


def test_trim_hook_caps_at_12_words() -> None:
    h = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    assert len(trim_hook(h).split()) == 12


def test_enforce_tags_dedupes_and_caps() -> None:
    tags = [f"tag{i}" for i in range(20)] + ["tag0", "#tag1"]
    out = enforce_tags(tags)
    assert len(out) == 15
    assert len(out) == len(set(t.lower() for t in out))
    assert all(not t.startswith("#") for t in out)


def test_enforce_hashtags_prefixes_and_caps() -> None:
    out = enforce_hashtags(["ai", "#football", "machine learning", *[f"h{i}" for i in range(10)]])
    assert len(out) <= 8
    assert all(h.startswith("#") for h in out)
    assert "#machinelearning" in out  # whitespace collapsed


def test_enforce_thumbnail_text_uppercases_and_caps_words() -> None:
    out = enforce_thumbnail_text(["this is way too many words", "big news"])
    assert out[0] == "THIS IS WAY TOO"  # capped at 4 words
    assert out[1] == "BIG NEWS"


def test_title_output_coerces_dict_items() -> None:
    """Smaller models sometimes return [{'title': 'X'}] instead of ['X']."""
    from app.agents.state import SEOOutput, TitleSmithOutput

    t = TitleSmithOutput(main_video_titles=[{"title": "Ronaldo Scores"}, "Plain"], per_clip=[])
    assert t.main_video_titles == ["Ronaldo Scores", "Plain"]
    s = SEOOutput(tags=[{"tag": "football"}, "ai"], hashtags=["#x"])
    assert s.tags == ["football", "ai"]


# --------------------------------------------------------------------------- #
# Fixtures for generation nodes
# --------------------------------------------------------------------------- #
def _approved() -> list[ApprovedClip]:
    c = ClipCandidate(
        clip_id="clip_1", start_s=0, end_s=30,
        transcript_text="Here is a self contained clip about expected goals in football.",
        reason_chosen="strong", scores=ClipScores(hook_potential=8, completeness=8, shareability=8),
    )
    return [ApprovedClip(candidate=c, verdict="approve", rank=1)]


def _state_with_map() -> ContentState:
    t = Transcript(
        video_id="v", language="en",
        segments=[Segment(start_s=i * 10, end_s=(i + 1) * 10, text=f"Point {i} about football and AI.")
                  for i in range(12)],
        total_duration_s=120,
    )
    return {
        "content_map": build_content_map(t),
        "approved_clips": _approved(),
        "node_trace": [], "errors": [], "token_usage": {},
    }


# --------------------------------------------------------------------------- #
# Title Smith: over-length title triggers ONE retry, comes back <= 60
# --------------------------------------------------------------------------- #
def test_title_retry_on_overlength(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    long_title = "X" * 80

    def fake(schema, messages, **kw):
        calls["n"] += 1
        if calls["n"] == 1:  # first response violates the 60-char cap
            return TitleSmithOutput(main_video_titles=[long_title, "short one"], per_clip=[]), USAGE
        return TitleSmithOutput(main_video_titles=["A reasonable title", "Another one"], per_clip=[]), USAGE

    monkeypatch.setattr(llm_mod, "structured_invoke", fake)
    out = ts_mod.title_smith_node(_state_with_map())

    assert calls["n"] == 2  # retry fired
    assert all(len(t) <= MAX_TITLE_CHARS for t in out["titles"].main_video_titles)


def test_title_backstop_trims_when_llm_keeps_violating(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(schema, messages, **kw):
        return TitleSmithOutput(main_video_titles=["Y" * 90], per_clip=[]), USAGE

    monkeypatch.setattr(llm_mod, "structured_invoke", fake)
    out = ts_mod.title_smith_node(_state_with_map())
    # Even though the LLM never complied, the hard trim guarantees <= 60.
    assert all(len(t) <= MAX_TITLE_CHARS for t in out["titles"].main_video_titles)


# --------------------------------------------------------------------------- #
# Hook Writer: hooks over 12 words get trimmed
# --------------------------------------------------------------------------- #
def test_hook_writer_trims_long_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    long_hook = "word " * 20
    out_obj = HooksOutput(clip_hooks=[HookSet(clip_id="clip_1", hooks=[
        Hook(style="curiosity", text=long_hook),
        Hook(style="bold_claim", text="A short punchy claim here"),
        Hook(style="question", text="Did you know this fact?"),
    ])])
    # Return the same (still-long) output twice: initial + retry. Backstop trims.
    monkeypatch.setattr(llm_mod, "structured_invoke", lambda s, m, **kw: (out_obj, USAGE))

    out = hw_mod.hook_writer_node(_state_with_map())
    hooks = out["hooks"]["clip_1"]
    assert all(len(h.text.split()) <= 12 for h in hooks)


def test_hook_writer_noop_without_clips() -> None:
    out = hw_mod.hook_writer_node({"approved_clips": [], "node_trace": []})
    assert out["hooks"] == {}


# --------------------------------------------------------------------------- #
# SEO Packager: tag/hashtag counts enforced
# --------------------------------------------------------------------------- #
def test_seo_enforces_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    seo_out = SEOOutput(
        description_md="Hook line\nSecond line\n\nSummary.\n\n00:00 Intro",
        tags=[f"tag{i}" for i in range(30)],
        hashtags=[f"tag{i}" for i in range(20)],
        shorts_caption="A short caption",
        reels_caption="A reels caption",
    )
    monkeypatch.setattr(llm_mod, "structured_invoke", lambda s, m, **kw: (seo_out, USAGE))

    out = seo_mod.seo_packager_node(_state_with_map())
    seo = out["seo"]
    assert len(seo.tags) <= 15
    assert len(seo.hashtags) <= 8
    assert all(h.startswith("#") for h in seo.hashtags)
