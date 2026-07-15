"""Tracing is optional and must fail open: no keys -> clean no-op.

Also proves (offline) that the critique loop is visible in the LangChain
callback stream that Langfuse consumes: a recording callback should see the
node spans for clip_scout / critic more than once when the loop fires.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler

from app.agents import llm as llm_mod
from app.agents.graph import build_graph
from app.agents.state import (
    ClipPick,
    ClipPickList,
    CriticNote,
    CriticReview,
    HooksOutput,
    SEOOutput,
    TitleSmithOutput,
)
from app.config import get_settings
from app.ingestion.chunker import build_content_map
from app.ingestion.transcript import Segment, Transcript
from app.observability import tracing


def test_disabled_returns_none_handler() -> None:
    # conftest sets no Langfuse keys, so tracing is disabled.
    assert get_settings().langfuse_enabled is False
    assert tracing.get_callback_handler("job_1") is None
    assert tracing.trace_config("job_1") == {}


def test_flush_noop_on_empty_config() -> None:
    # Must not raise even with nothing to flush.
    tracing.flush({})
    tracing.flush({"callbacks": []})


def test_enabled_builds_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    get_settings.cache_clear()

    assert get_settings().langfuse_enabled is True
    cfg = tracing.trace_config("job_42")
    # Either a fully wired config, or {} if the SDK failed to init — never raises.
    assert isinstance(cfg, dict)
    if cfg:
        assert cfg["metadata"]["langfuse_session_id"] == "job_42"
        assert cfg["callbacks"]
    tracing.flush(cfg)  # must be safe regardless


class _NodeRecorder(BaseCallbackHandler):
    """Records chain (node) span names — a proxy for what Langfuse traces."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name")
        if name:
            self.names.append(name)


USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "cost_usd": 0.0}


def test_critique_loop_visible_in_callback_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake(schema, messages, **kw):
        if schema is ClipPickList:
            return ClipPickList(clips=[
                ClipPick(start_s=0, end_s=30, transcript_text="A complete thought.",
                         reason_chosen="x", hook_potential=8, completeness=8, shareability=8),
            ]), USAGE
        if schema is CriticReview:
            calls["n"] += 1
            verdict = "reject" if calls["n"] == 1 else "approve"
            return CriticReview(notes=[CriticNote(clip_id="clip_1", verdict=verdict, reasons=["r"])]), USAGE
        if schema is HooksOutput:
            return HooksOutput(clip_hooks=[]), USAGE
        if schema is TitleSmithOutput:
            return TitleSmithOutput(main_video_titles=["T"], per_clip=[]), USAGE
        if schema is SEOOutput:
            return SEOOutput(description_md="d", tags=[], hashtags=[], shorts_caption="s", reels_caption="r"), USAGE
        raise AssertionError(schema)

    monkeypatch.setattr(llm_mod, "structured_invoke", fake)

    t = Transcript(video_id="v", language="en",
                   segments=[Segment(start_s=i * 10, end_s=(i + 1) * 10, text=f"Line {i}.") for i in range(40)],
                   total_duration_s=400)
    state = {"transcript": t, "content_map": build_content_map(t), "content_type": "unknown",
             "node_trace": [], "errors": [], "token_usage": {}}

    recorder = _NodeRecorder()
    build_graph(max_critic_rounds=2).invoke(state, config={"callbacks": [recorder]})

    # The loop fired -> clip_scout and critic each show up as spans more than once.
    assert recorder.names.count("clip_scout") == 2
    assert recorder.names.count("critic") == 2
    assert "select_clips" in recorder.names
