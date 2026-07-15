"""Manual Phase-5 smoke: run the WHOLE graph (clips + hooks + titles + SEO).

Usage:
    python scripts/demo_full.py <youtube_url | transcript_file>
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
        print("usage: python scripts/demo_full.py <youtube_url | transcript_file>")
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
    print(f"  {len(content_map.blocks)} blocks, {seconds_to_hhmmss(transcript.total_duration_s)}\n")

    initial: ContentState = {
        "video_url": source, "transcript": transcript, "content_map": content_map,
        "content_type": "unknown", "node_trace": [], "errors": [], "token_usage": {},
    }
    print("→ Running full pipeline...\n")
    final = build_graph(max_critic_rounds=settings.max_critic_rounds).invoke(initial)

    approved = final.get("approved_clips", [])
    hooks = final.get("hooks", {})
    titles = final.get("titles")
    seo = final.get("seo")

    print("=" * 70)
    print(f"APPROVED CLIPS: {len(approved)}  |  critique rounds: {final.get('critic_round', 0)}")
    print("=" * 70)
    for a in approved:
        c = a.candidate
        print(f"\n#{a.rank} [{c.clip_id}] {seconds_to_hhmmss(c.start_s)}–{seconds_to_hhmmss(c.end_s)} "
              f"— {c.total_score}/30")
        print(f"   text: {c.transcript_text[:110]}")
        for h in hooks.get(c.clip_id, []):
            print(f"   hook ({h.style}): {h.text}")
        if titles and c.clip_id in titles.per_clip:
            ts = titles.per_clip[c.clip_id]
            print(f"   titles: {ts.titles}")
            print(f"   thumbnail: {ts.thumbnail_text}")

    if titles:
        print("\n" + "=" * 70)
        print("MAIN VIDEO TITLES")
        print("=" * 70)
        for t in titles.main_video_titles:
            print(f"  • {t}  ({len(t)} chars)")

    if seo:
        print("\n" + "=" * 70)
        print("SEO PACK")
        print("=" * 70)
        print(seo.description_md)
        print(f"\nTAGS ({len(seo.tags)}): {', '.join(seo.tags)}")
        print(f"HASHTAGS ({len(seo.hashtags)}): {' '.join(seo.hashtags)}")
        print(f"\nSHORTS CAPTION: {seo.shorts_caption}")
        print(f"REELS CAPTION: {seo.reels_caption}")

    usage = final.get("token_usage", {})
    total_tokens = sum(u.get("total_tokens", 0) for u in usage.values())
    total_cost = sum(u.get("cost_usd", 0.0) for u in usage.values())
    print("\n" + "=" * 70)
    print(f"COST: {total_tokens} tokens across {len(usage)} LLM calls, ~${total_cost:.4f}")
    if final.get("errors"):
        print("warnings:", final["errors"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
