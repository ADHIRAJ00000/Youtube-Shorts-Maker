"""Eval harness: clip-selection recall, critic ON/OFF ablation, LLM-as-judge.

Usage:
    python evals/run_evals.py                 # all fixtures in golden_set.json
    python evals/run_evals.py --fixtures aircAruvnKk
    python evals/run_evals.py --no-judge      # skip the LLM-as-judge pass

Metrics
  * Recall: fraction of golden clips an agent clip covers by >=50% overlap.
  * Ablation: recall with the critic ON (full loop) vs OFF (approve-everything),
    the headline "did the self-critique loop improve clip selection" number.
  * Judge: a separate LLM rates hooks/titles 1-10 for truthfulness + strength.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import END, START, StateGraph  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.agents import llm  # noqa: E402
from app.agents.clip_scout import clip_scout_node  # noqa: E402
from app.agents.coordinator import coordinator_node  # noqa: E402
from app.agents.critic import critic_node  # noqa: E402
from app.agents.graph import _resilient, build_graph, make_route_after_critic  # noqa: E402
from app.agents.select_clips import select_clips_node  # noqa: E402
from app.agents.state import ContentState, CriticNote  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.ingestion.chunker import build_content_map  # noqa: E402
from app.ingestion.transcript import load_transcript_file  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.json"


# --------------------------------------------------------------------------- #
# Pure metric helpers (unit-tested)
# --------------------------------------------------------------------------- #
def intersection_s(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def clip_covers_golden(clip: dict, golden: dict, min_overlap: float = 0.5) -> bool:
    """True if `clip` overlaps >= min_overlap of the golden clip's duration."""
    gdur = golden["end_s"] - golden["start_s"]
    if gdur <= 0:
        return False
    inter = intersection_s(clip["start_s"], clip["end_s"], golden["start_s"], golden["end_s"])
    return inter >= min_overlap * gdur


def golden_recall(clips: list[dict], golden_clips: list[dict]) -> dict[str, Any]:
    """Fraction of golden clips covered by at least one agent clip."""
    hits = sum(1 for g in golden_clips if any(clip_covers_golden(c, g) for c in clips))
    total = len(golden_clips)
    return {"hits": hits, "total": total, "recall": (hits / total) if total else 0.0}


def selection_precision(clips: list[dict], golden_clips: list[dict]) -> float:
    """Fraction of agent clips that land on a golden clip.

    The complement of recall's blind spot: recall rewards selecting *more*
    clips, precision rewards selecting the *right* ones. The critic trades the
    former for the latter, so reporting both keeps the ablation honest.
    """
    if not clips:
        return 0.0
    good = sum(1 for c in clips if any(clip_covers_golden(c, g) for g in golden_clips))
    return good / len(clips)


# --------------------------------------------------------------------------- #
# Clips-only sub-graph (token-efficient — no hooks/titles/SEO)
# --------------------------------------------------------------------------- #
def _approve_all_critic(state: ContentState) -> dict:
    round_ = state.get("critic_round", 0) + 1
    notes = [CriticNote(clip_id=c.clip_id, verdict="approve", reasons=["critic disabled"])
             for c in state.get("clip_candidates", []) or []]
    return {"critic_feedback": notes, "critic_round": round_, "node_trace": ["critic"]}


def build_clips_graph(max_rounds: int, critic_enabled: bool):
    g: StateGraph = StateGraph(ContentState)
    g.add_node("coordinator", _resilient("coordinator", coordinator_node))
    g.add_node("clip_scout", _resilient("clip_scout", clip_scout_node))
    critic_fn = critic_node if critic_enabled else _approve_all_critic
    g.add_node("critic", _resilient("critic", critic_fn))
    g.add_node("select_clips", _resilient("select_clips", select_clips_node))
    g.add_edge(START, "coordinator")
    g.add_edge("coordinator", "clip_scout")  # force the clip pipeline for eval
    g.add_edge("clip_scout", "critic")
    g.add_conditional_edges(
        "critic",
        make_route_after_critic(max_rounds if critic_enabled else 0),
        {"clip_scout": "clip_scout", "select_clips": "select_clips"},
    )
    g.add_edge("select_clips", END)
    return g.compile()


def _approved_dicts(final: ContentState) -> list[dict]:
    return [
        {"clip_id": a.candidate.clip_id, "start_s": a.candidate.start_s,
         "end_s": a.candidate.end_s, "total_score": a.candidate.total_score}
        for a in (final.get("approved_clips", []) or [])
    ]


def run_clip_selection(state: ContentState, critic_enabled: bool) -> list[dict]:
    settings = get_settings()
    graph = build_clips_graph(settings.max_critic_rounds, critic_enabled)
    final = graph.invoke(dict(state))
    return _approved_dicts(final)


# --------------------------------------------------------------------------- #
# LLM-as-judge
# --------------------------------------------------------------------------- #
class JudgeScores(BaseModel):
    # 0-10: the judge must be able to score 0 for a fully untruthful hook/title.
    hook_truthfulness: int = Field(ge=0, le=10, description="Are hooks supported by clip content?")
    hook_strength: int = Field(ge=0, le=10, description="Would the hooks stop a scroll?")
    title_truthfulness: int = Field(ge=0, le=10, description="Are titles honest to content?")
    title_ctr_potential: int = Field(ge=0, le=10, description="Would titles earn clicks?")
    rationale: str = ""


_JUDGE_SYSTEM = (
    "You are a strict content-quality judge. Given clips with their hooks and "
    "titles, rate 1-10 on: hook_truthfulness (supported by the clip text), "
    "hook_strength (scroll-stopping), title_truthfulness (honest, no bait), and "
    "title_ctr_potential. Be critical. Output ONLY the structured schema."
)


def judge_generation(final: ContentState) -> Optional[JudgeScores]:
    approved = final.get("approved_clips", []) or []
    hooks = final.get("hooks", {}) or {}
    titles = final.get("titles")
    if not approved:
        return None
    blocks = []
    for a in approved:
        c = a.candidate
        hs = "; ".join(f"[{h.style}] {h.text}" for h in hooks.get(c.clip_id, []))
        ts = ", ".join(titles.per_clip[c.clip_id].titles) if titles and c.clip_id in titles.per_clip else ""
        blocks.append(f'CLIP "{c.transcript_text[:200]}"\n  hooks: {hs}\n  titles: {ts}')
    main = ", ".join(titles.main_video_titles) if titles else ""
    human = f"MAIN VIDEO TITLES: {main}\n\n" + "\n\n".join(blocks)
    try:
        scores, _ = llm.structured_invoke(
            JudgeScores,
            [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=human)],
            temperature=0.0,
        )
        return scores
    except Exception as exc:  # noqa: BLE001
        print(f"    (judge failed: {exc})")
        return None


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _load_state(video_id: str) -> ContentState:
    transcript = load_transcript_file(FIXTURES_DIR / f"{video_id}.json")
    return {
        "video_url": video_id,
        "transcript": transcript,
        "content_map": build_content_map(transcript),
        "content_type": "unknown",
        "node_trace": [], "errors": [], "token_usage": {},
    }


def run(fixtures_filter: Optional[list[str]], do_judge: bool) -> None:
    golden = json.loads(GOLDEN_PATH.read_text())
    entries = golden["fixtures"]
    if fixtures_filter:
        entries = [e for e in entries if e["fixture"] in fixtures_filter]

    rows: list[dict] = []
    for e in entries:
        vid = e["video_id"]
        print(f"\n=== {vid} — {e.get('title', '')} ===")
        if not (FIXTURES_DIR / f"{vid}.json").exists():
            print(f"  ! fixture missing: run  python evals/make_fixtures.py {vid}")
            continue
        try:
            base = _load_state(vid)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! failed to load fixture: {exc}")
            continue

        # Clip-selection recall uses the cheap clips-only graph for BOTH modes,
        # so the ablation is reliable even under rate limits.
        print("  running critic ON (self-critique loop)...")
        clips_on = run_clip_selection(base, critic_enabled=True)
        print("  running critic OFF (approve-all)...")
        clips_off = run_clip_selection(base, critic_enabled=False)

        rec_on = golden_recall(clips_on, e["golden_clips"])
        rec_off = golden_recall(clips_off, e["golden_clips"])
        prec_on = selection_precision(clips_on, e["golden_clips"])
        prec_off = selection_precision(clips_off, e["golden_clips"])

        judge = None
        if do_judge:
            print("  running full pipeline for LLM-as-judge...")
            try:
                full = build_graph(max_critic_rounds=get_settings().max_critic_rounds).invoke(dict(base))
                judge = judge_generation(full)
            except Exception as exc:  # noqa: BLE001
                print(f"    (judge pipeline failed: {exc})")

        rows.append({
            "fixture": vid,
            "recall_on": rec_on, "recall_off": rec_off,
            "prec_on": prec_on, "prec_off": prec_off,
            "clips_on": len(clips_on), "clips_off": len(clips_off),
            "judge": judge,
        })
        print(f"  recall ON={rec_on['recall']:.0%} / OFF={rec_off['recall']:.0%} | "
              f"precision ON={prec_on:.0%} / OFF={prec_off:.0%} | "
              f"clips ON={len(clips_on)} OFF={len(clips_off)}")

    _print_scorecard(rows, do_judge)


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _print_scorecard(rows: list[dict], do_judge: bool) -> None:
    if not rows:
        print("\nNo fixtures evaluated.")
        return
    print("\n" + "=" * 74)
    print("SCORECARD")
    print("=" * 74)
    print(f"{'fixture':<16}{'rec ON':>9}{'rec OFF':>9}{'prec ON':>9}"
          f"{'prec OFF':>10}{'clips ON/OFF':>15}")
    print("-" * 74)
    for r in rows:
        clips = f"{r['clips_on']}/{r['clips_off']}"
        print(f"{r['fixture']:<16}{r['recall_on']['recall']:>8.0%}"
              f"{r['recall_off']['recall']:>9.0%}{r['prec_on']:>9.0%}"
              f"{r['prec_off']:>10.0%}{clips:>15}")

    rec_on = _avg([r["recall_on"]["recall"] for r in rows])
    rec_off = _avg([r["recall_off"]["recall"] for r in rows])
    prec_on = _avg([r["prec_on"] for r in rows])
    prec_off = _avg([r["prec_off"] for r in rows])
    print("-" * 74)
    print(f"{'AVERAGE':<16}{rec_on:>8.0%}{rec_off:>9.0%}{prec_on:>9.0%}{prec_off:>10.0%}")
    print("=" * 74)

    def _verb(on: float, off: float) -> str:
        return "improved" if on > off else ("reduced" if on < off else "held")

    print(f"\n>>> Critic ablation:")
    print(f"    recall    {_verb(rec_on, rec_off)}: {rec_off:.0%} (off) -> {rec_on:.0%} (on)")
    print(f"    precision {_verb(prec_on, prec_off)}: {prec_off:.0%} (off) -> {prec_on:.0%} (on)")
    print("    (Recall favors selecting more clips; precision favors selecting the "
          "right ones. The critic trades quantity for quality.)")

    if do_judge:
        judged = [r["judge"] for r in rows if r["judge"]]
        if judged:
            print("\nLLM-as-judge (avg 1-10):")
            print(f"  hook truthfulness : {_avg([j.hook_truthfulness for j in judged]):.1f}")
            print(f"  hook strength     : {_avg([j.hook_strength for j in judged]):.1f}")
            print(f"  title truthfulness: {_avg([j.title_truthfulness for j in judged]):.1f}")
            print(f"  title CTR potential: {_avg([j.title_ctr_potential for j in judged]):.1f}")
    print("\nNOTE: golden clips are PLACEHOLDERS until you mark your own — see golden_set.json.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", nargs="*", help="fixture ids to run (default: all)")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge pass")
    args = ap.parse_args()
    run(args.fixtures, do_judge=not args.no_judge)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
