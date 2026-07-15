"""Unit tests for ingestion: URL parsing, file loading, chunking, timestamp math.

None of these hit the network — YouTube fetching is covered by the CLI smoke
check in the phase's Definition of Done.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.chunker import build_content_map
from app.ingestion.transcript import (
    Segment,
    Transcript,
    TranscriptError,
    extract_video_id,
    load_transcript_file,
)
from app.tools.video_tools import (
    clamp_to_bounds,
    clip_duration,
    overlap_fraction,
    overlaps,
    seconds_to_hhmmss,
)

FIXTURES = Path(__file__).resolve().parent.parent / "evals" / "fixtures"


# --------------------------------------------------------------------------- #
# URL / id extraction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/embed/dQw4w9WgXcQ?start=1", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_video_id(url: str, expected: str) -> None:
    assert extract_video_id(url) == expected


def test_extract_video_id_invalid() -> None:
    with pytest.raises(TranscriptError):
        extract_video_id("https://example.com/not-a-video")


def test_proxy_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.ingestion.transcript import _build_proxy_config

    # No proxy env -> None.
    for k in ("YOUTUBE_PROXY_USERNAME", "YOUTUBE_PROXY_PASSWORD",
              "YOUTUBE_HTTP_PROXY", "YOUTUBE_HTTPS_PROXY"):
        monkeypatch.delenv(k, raising=False)
    assert _build_proxy_config() is None

    # Webshare creds -> a proxy config object.
    monkeypatch.setenv("YOUTUBE_PROXY_USERNAME", "user")
    monkeypatch.setenv("YOUTUBE_PROXY_PASSWORD", "pass")
    assert _build_proxy_config() is not None


# --------------------------------------------------------------------------- #
# File loading
# --------------------------------------------------------------------------- #
def test_load_srt_file() -> None:
    t = load_transcript_file(FIXTURES / "sample.srt")
    assert t.source == "file"
    assert len(t.segments) == 5
    assert t.segments[0].start_s == 0.0
    assert t.segments[0].end_s == 4.0
    assert "Alpha Zone" in t.segments[0].text
    assert t.total_duration_s == pytest.approx(27.5)


def test_load_txt_file_estimates_timestamps(tmp_path: Path) -> None:
    p = tmp_path / "plain.txt"
    p.write_text("This is one sentence. Here is a second one! And a third?")
    t = load_transcript_file(p)
    assert t.source == "estimated"
    assert len(t.segments) == 3
    # Timestamps should be monotonically increasing.
    assert t.segments[0].start_s == 0.0
    assert t.segments[1].start_s == t.segments[0].end_s


def test_load_json_ytapi_form(tmp_path: Path) -> None:
    p = tmp_path / "t.json"
    p.write_text('[{"text": "hello", "start": 0.0, "duration": 2.0}, '
                 '{"text": "world", "start": 2.0, "duration": 2.5}]')
    t = load_transcript_file(p)
    assert len(t.segments) == 2
    assert t.segments[1].end_s == pytest.approx(4.5)


def test_load_missing_file() -> None:
    with pytest.raises(TranscriptError):
        load_transcript_file("/nope/does_not_exist.srt")


def test_unsupported_file_type(tmp_path: Path) -> None:
    p = tmp_path / "bad.md"
    p.write_text("nope")
    with pytest.raises(TranscriptError):
        load_transcript_file(p)


# --------------------------------------------------------------------------- #
#  Chunker edge cases
# --------------------------------------------------------------------------- #
def _make_transcript(segments: list[Segment]) -> Transcript:
    total = segments[-1].end_s if segments else 0.0
    return Transcript(
        video_id="test",
        language="en",
        segments=segments,
        total_duration_s=total,
    )


def test_chunker_empty() -> None:
    t = _make_transcript([])
    cmap = build_content_map(t)
    assert cmap.blocks == []
    assert cmap.total_words == 0


def test_chunker_very_short() -> None:
    t = _make_transcript(
        [Segment(start_s=0, end_s=5, text="A single short thought here.")]
    )
    cmap = build_content_map(t)
    assert len(cmap.blocks) == 1
    assert cmap.blocks[0].word_count == 5
    assert cmap.blocks[0].start_s == 0.0


def test_chunker_merges_into_target_blocks() -> None:
    # 20 segments of 5s each = 100s total; target ~45s should yield ~2-3 blocks.
    segs = []
    for i in range(20):
        # End every 9th segment on a sentence boundary.
        text = f"segment number {i} words here"
        if i % 9 == 8:
            text += "."
        segs.append(Segment(start_s=i * 5, end_s=(i + 1) * 5, text=text))
    cmap = build_content_map(_make_transcript(segs))
    assert 1 <= len(cmap.blocks) <= 4
    # Blocks are contiguous and ordered.
    for a, b in zip(cmap.blocks, cmap.blocks[1:]):
        assert b.index == a.index + 1
        assert b.start_s >= a.start_s
    # No block wildly exceeds the hard max (60s) by more than one segment.
    assert all(blk.duration_s <= 65 for blk in cmap.blocks)


def test_chunker_very_long_goes_coarse() -> None:
    # ~25k words -> exceeds the (raised) default token budget -> coarse map.
    segs = []
    for i in range(5000):
        segs.append(
            Segment(start_s=i * 3, end_s=(i + 1) * 3, text="five little words go here.")
        )
    cmap = build_content_map(_make_transcript(segs))
    assert cmap.is_coarse is True
    assert cmap.estimated_tokens > 30_000
    assert len(cmap.blocks) >= 1


def test_render_for_prompt_has_timestamps() -> None:
    t = _make_transcript(
        [Segment(start_s=0, end_s=10, text="Hello there world.")]
    )
    rendered = build_content_map(t).render_for_prompt()
    assert "block 0" in rendered
    assert "0.0s-10.0s" in rendered


# --------------------------------------------------------------------------- #
# Timestamp utilities
# --------------------------------------------------------------------------- #
def test_seconds_to_hhmmss() -> None:
    assert seconds_to_hhmmss(0) == "00:00"
    assert seconds_to_hhmmss(65) == "01:05"
    assert seconds_to_hhmmss(3661) == "01:01:01"


def test_clip_duration() -> None:
    assert clip_duration(10, 40) == 30.0
    assert clip_duration(40, 10) == 0.0  # clamped


def test_overlaps_and_fraction() -> None:
    assert overlaps(0, 10, 5, 15) is True
    assert overlaps(0, 10, 10, 20) is False
    assert overlap_fraction(0, 10, 5, 15) == pytest.approx(0.5)
    assert overlap_fraction(0, 10, 100, 110) == 0.0


def test_clamp_to_bounds() -> None:
    assert clamp_to_bounds(-5, 200, 0, 100) == (0.0, 100.0)
    assert clamp_to_bounds(90, 80, 0, 100) == (80.0, 90.0)  # reordered
