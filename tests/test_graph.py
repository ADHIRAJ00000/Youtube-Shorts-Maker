"""Graph-topology tests: node order, the critique loop, short-form skip.

These inject pure fake nodes for clip_scout/critic (no LLM) — the point is to
prove the *wiring* is correct. Real agent behavior is covered by
test_critic_loop.py with a mocked LLM.
"""

from __future__ import annotations

from app.agents.graph import build_graph
from app.agents.state import (
    ClipCandidate,
    ClipScores,
    ContentState,
    CriticNote,
)
from app.ingestion.transcript import Segment, Transcript


def _transcript(duration_s: float) -> Transcript:
    return Transcript(
        video_id="vid",
        language="en",
        segments=[Segment(start_s=0, end_s=duration_s, text="hello world.")],
        total_duration_s=duration_s,
    )


def _initial(duration_s: float) -> ContentState:
    return {
        "transcript": _transcript(duration_s),
        "content_type": "unknown",
        "node_trace": [],
        "errors": [],
        "token_usage": {},
    }


def _count(trace: list[str], name: str) -> int:
    return sum(1 for n in trace if n == name)


def _candidate(cid: str, score: int = 8) -> ClipCandidate:
    return ClipCandidate(
        clip_id=cid,
        start_s=0,
        end_s=30,
        transcript_text="a self contained thought that ends cleanly.",
        reason_chosen="strong hook",
        scores=ClipScores(hook_potential=score, completeness=score, shareability=score),
    )


def _fake_scout(state: ContentState) -> dict:
    return {
        "clip_candidates": [_candidate("clip_1", 9), _candidate("clip_2", 7)],
        "node_trace": ["clip_scout"],
    }


def _gen_node(name: str):
    """Trivial fake for a generation node — records trace, no LLM."""
    def _n(state: ContentState) -> dict:
        return {"node_trace": [name]}
    return _n


# Generation-node fakes so topology tests never touch the LLM.
_GEN_FAKES = {
    "hook_writer": _gen_node("hook_writer"),
    "title_smith": _gen_node("title_smith"),
    "seo_packager": _gen_node("seo_packager"),
}


# --------------------------------------------------------------------------- #
# 1) Full path — happy order
# --------------------------------------------------------------------------- #
def test_full_path_node_order() -> None:
    def approve_all(state: ContentState) -> dict:
        round_ = state.get("critic_round", 0) + 1
        notes = [CriticNote(clip_id=c.clip_id, verdict="approve", reasons=["good"])
                 for c in state["clip_candidates"]]
        return {"critic_feedback": notes, "critic_round": round_, "node_trace": ["critic"]}

    graph = build_graph(node_overrides={"clip_scout": _fake_scout, "critic": approve_all, **_GEN_FAKES})
    final = graph.invoke(_initial(600))
    trace = final["node_trace"]
    print("FULL PATH:", trace)

    assert trace[0] == "coordinator"
    assert trace.index("clip_scout") < trace.index("critic")
    assert trace.index("critic") < trace.index("select_clips")
    assert trace.index("select_clips") < trace.index("seo_packager")
    assert "hook_writer" in trace and "title_smith" in trace
    assert trace.index("hook_writer") < trace.index("seo_packager")
    assert trace.index("title_smith") < trace.index("seo_packager")
    assert trace.index("seo_packager") < trace.index("assembler")
    assert _count(trace, "clip_scout") == 1  # no loop
    # Approval gate ranked both, best score first.
    approved = final["approved_clips"]
    assert [a.candidate.clip_id for a in approved] == ["clip_1", "clip_2"]
    assert approved[0].rank == 1


# --------------------------------------------------------------------------- #
# 2) Critique loop fires exactly once on a reject, then proceeds
# --------------------------------------------------------------------------- #
def test_critique_loop_fires_once() -> None:
    def fake_critic(state: ContentState) -> dict:
        round_ = state.get("critic_round", 0) + 1
        if round_ == 1:
            fb = [CriticNote(clip_id="clip_1", verdict="reject", reasons=["mid-thought cutoff"])]
        else:
            fb = [CriticNote(clip_id="clip_1", verdict="approve", reasons=["fixed"]),
                  CriticNote(clip_id="clip_2", verdict="approve", reasons=["ok"])]
        return {"critic_feedback": fb, "critic_round": round_, "node_trace": ["critic"]}

    graph = build_graph(max_critic_rounds=2,
                        node_overrides={"clip_scout": _fake_scout, "critic": fake_critic, **_GEN_FAKES})
    final = graph.invoke(_initial(600))
    trace = final["node_trace"]
    print("LOOP-ONCE:", trace)

    assert _count(trace, "clip_scout") == 2  # looped once
    assert _count(trace, "critic") == 2
    assert final["critic_round"] == 2
    assert "select_clips" in trace and "assembler" in trace


# --------------------------------------------------------------------------- #
# 3) Loop terminates at MAX_CRITIC_ROUNDS; fallback keeps pipeline alive
# --------------------------------------------------------------------------- #
def test_critique_loop_terminates_at_max() -> None:
    def always_reject(state: ContentState) -> dict:
        round_ = state.get("critic_round", 0) + 1
        fb = [CriticNote(clip_id=c.clip_id, verdict="reject", reasons=["weak"])
              for c in state["clip_candidates"]]
        return {"critic_feedback": fb, "critic_round": round_, "node_trace": ["critic"]}

    graph = build_graph(max_critic_rounds=2,
                        node_overrides={"clip_scout": _fake_scout, "critic": always_reject, **_GEN_FAKES})
    final = graph.invoke(_initial(600))
    trace = final["node_trace"]
    print("LOOP-MAX:", trace)

    assert _count(trace, "critic") == 2
    assert _count(trace, "clip_scout") == 2
    assert final["critic_round"] == 2
    # Everything rejected -> top-2 fallback -> pipeline still completes.
    assert len(final["approved_clips"]) == 2
    assert any("fell back" in e for e in final["errors"])
    assert "assembler" in trace


# --------------------------------------------------------------------------- #
# 4) Short-form video skips the clip pipeline entirely
# --------------------------------------------------------------------------- #
def test_short_video_skips_clip_pipeline() -> None:
    graph = build_graph()
    final = graph.invoke(_initial(60))  # under 180s, no LLM nodes reached
    trace = final["node_trace"]
    print("SHORT PATH:", trace)

    assert final["content_type"] == "short_form"
    assert "clip_scout" not in trace
    assert "critic" not in trace
    assert "select_clips" not in trace
    assert "title_smith" in trace
    assert "seo_packager" in trace
    assert "assembler" in trace


# --------------------------------------------------------------------------- #
# 5) A failing node degrades the run instead of crashing it
# --------------------------------------------------------------------------- #
def test_failing_node_degrades_gracefully() -> None:
    def boom(state: ContentState) -> dict:
        raise RuntimeError("simulated agent failure")

    graph = build_graph(node_overrides={
        "clip_scout": _fake_scout,
        "critic": lambda s: {
            "critic_feedback": [CriticNote(clip_id=c.clip_id, verdict="approve", reasons=["ok"])
                                for c in s["clip_candidates"]],
            "critic_round": s.get("critic_round", 0) + 1,
            "node_trace": ["critic"],
        },
        "hook_writer": boom,  # this agent blows up
        "title_smith": _gen_node("title_smith"),
        "seo_packager": _gen_node("seo_packager"),
    })
    final = graph.invoke(_initial(600))

    # Pipeline still completed to the assembler...
    assert "assembler" in final["node_trace"]
    # ...and the failure was recorded, not raised.
    assert any("hook_writer failed" in e for e in final["errors"])
