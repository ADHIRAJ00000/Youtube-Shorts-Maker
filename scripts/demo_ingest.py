"""Manual Phase-2 smoke: ingest a source into Transcript + ContentMap.

Usage:
    python scripts/demo_ingest.py                      # uses the .srt fixture
    python scripts/demo_ingest.py <youtube_url|file>   # any source
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.chunker import build_content_map  # noqa: E402
from app.ingestion.transcript import (  # noqa: E402
    TranscriptError,
    load_transcript,
)
from app.tools.video_tools import seconds_to_hhmmss  # noqa: E402

DEFAULT = str(
    Path(__file__).resolve().parent.parent / "evals" / "fixtures" / "sample.srt"
)


def main() -> int:
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    print(f"→ Ingesting: {source}\n")
    try:
        transcript = load_transcript(source)
    except TranscriptError as exc:
        print(f"❌ TranscriptError (clean, typed): {exc}")
        return 1

    print(
        f"Transcript: video_id={transcript.video_id} lang={transcript.language} "
        f"source={transcript.source} segments={len(transcript.segments)} "
        f"duration={seconds_to_hhmmss(transcript.total_duration_s)} "
        f"words={transcript.word_count}"
    )
    if transcript.available_languages:
        print(f"Available languages: {transcript.available_languages}")

    cmap = build_content_map(transcript)
    print(
        f"\nContentMap: {len(cmap.blocks)} blocks | "
        f"~{cmap.estimated_tokens} tokens | coarse={cmap.is_coarse}\n"
    )
    for b in cmap.blocks:
        rng = f"{seconds_to_hhmmss(b.start_s)}–{seconds_to_hhmmss(b.end_s)}"
        preview = b.text if len(b.text) <= 90 else b.text[:87] + "..."
        print(f"  [block {b.index}] {rng} ({b.word_count}w): {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
