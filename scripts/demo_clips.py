"""Manual Phase-4 smoke: run the real Clip Scout + Critic loop on a video.

Usage:
    python scripts/demo_clips.py <youtube_url | transcript_file>

Requires a real LLM_API_KEY in .env (free Groq key: https://console.groq.com/keys).
Prints approved clips (timestamps, reasons, scores), rejected clips with the
critic's reasons, the number of critique rounds, and token/cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.graph import build_graph  # noqa: E402
from app.agents.state import ContentState  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.ingestion.chunker import build_content_map  # noqa: E402
from app.ingestion.transcript import TranscriptError, load_transcript  # noqa: E402
from app.tools.video_tools import seconds_to_hhmmss  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/demo_clips.py <youtube_url | transcript_file>")
        return 2
    source = sys.argv[1]
    settings = get_settings()

    print(f"→ Ingesting: {source}")
    try:
        transcript = load_transcript(source)
    except TranscriptError as exc:
        print(f"❌ {exc}")
        return 1
    content_map = build_content_map(transcript)
    print(
        f"  {len(transcript.segments)} segments -> {len(content_map.blocks)} blocks "
        f"({seconds_to_hhmmss(transcript.total_duration_s)}, ~{content_map.estimated_tokens} tokens)\n"
    )

    initial: ContentState = {
        "video_url": source,
        "transcript": transcript,
        "content_map": content_map,
        "content_type": "unknown",
        "node_trace": [],
        "errors": [],
        "token_usage": {},
    }
    graph = build_graph(max_critic_rounds=settings.max_critic_rounds)
    print("→ Running clip pipeline (Clip Scout → Critic loop → select)...\n")
    final = graph.invoke(initial)

    feedback = {n.clip_id: n for n in final.get("critic_feedback", [])}
    approved = final.get("approved_clips", [])
    rounds = final.get("critic_round", 0)

    print(f"=== APPROVED CLIPS ({len(approved)}) — {rounds} critique round(s) ===")
    for a in approved:
        c = a.candidate
        rng = f"{seconds_to_hhmmss(c.start_s)}–{seconds_to_hhmmss(c.end_s)}"
        print(f"\n#{a.rank} [{c.clip_id}] {rng} ({c.duration_s:.0f}s) — score {c.total_score}/30")
        print(f"   scores: hook={c.scores.hook_potential} complete={c.scores.completeness} share={c.scores.shareability}")
        print(f"   why: {c.reason_chosen}")
        print(f"   text: {c.transcript_text[:140]}")

    proposed_ids = {c.clip_id for c in final.get("clip_candidates", [])}
    approved_ids = {a.candidate.clip_id for a in approved}
    rejected_ids = proposed_ids - approved_ids
    if rejected_ids:
        print(f"\n=== REJECTED / DROPPED ({len(rejected_ids)}) ===")
        for cid in sorted(rejected_ids):
            note = feedback.get(cid)
            if note:
                print(f"  [{cid}] {note.verdict.upper()}: {'; '.join(note.reasons)}")

    usage = final.get("token_usage", {})
    total_tokens = sum(u.get("total_tokens", 0) for u in usage.values())
    total_cost = sum(u.get("cost_usd", 0.0) for u in usage.values())
    print(f"\n=== COST === {total_tokens} tokens across {len(usage)} calls, ~${total_cost:.4f}")
    if final.get("errors"):
        print("warnings:", final["errors"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
