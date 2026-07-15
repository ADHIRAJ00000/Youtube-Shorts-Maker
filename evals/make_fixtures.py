"""Save YouTube transcripts as offline eval fixtures.

Usage:
    python evals/make_fixtures.py <video_id_or_url> [<video_id_or_url> ...]

Then add a matching entry (with YOUR golden clip ranges) to golden_set.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.transcript import TranscriptError, fetch_transcript  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for src in sys.argv[1:]:
        try:
            t = fetch_transcript(src)
        except TranscriptError as exc:
            print(f"✗ {src}: {exc}")
            continue
        data = {
            "video_id": t.video_id,
            "language": t.language,
            "total_duration_s": t.total_duration_s,
            "segments": [{"start_s": s.start_s, "end_s": s.end_s, "text": s.text} for s in t.segments],
        }
        path = FIXTURES / f"{t.video_id}.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"✓ saved {path.name}: {len(t.segments)} segments, {t.total_duration_s:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
