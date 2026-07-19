"""Tests for audience-retention ingestion and clip fusion.

All pure-function tests — no network. The heatmap is synthesized so peak
detection, ranking, and suppression are checked against known shapes.
"""

from __future__ import annotations

import pytest

from app.agents.clip_scout import _fuse_heatmap, _lift_to_scores, _text_between
from app.agents.state import ClipCandidate, ClipScores
from app.ingestion.heatmap import MAX_PEAK_WINDOW_S, Heatmap, HeatmapBucket
from app.ingestion.transcript import Segment, Transcript


def _heatmap(values: list[float], bucket_s: float = 10.0) -> Heatmap:
    """Build a heatmap from a list of bucket values."""
    return Heatmap(
        video_id="vid",
        buckets=[
            HeatmapBucket(start_s=i * bucket_s, end_s=(i + 1) * bucket_s, value=v)
            for i, v in enumerate(values)
        ],
    )


def _transcript(duration_s: float = 100.0) -> Transcript:
    segs = [
        Segment(start_s=float(i), end_s=float(i + 10), text=f"words at {i}")
        for i in range(0, int(duration_s), 10)
    ]
    return Transcript(
        video_id="vid", language="en", segments=segs, total_duration_s=duration_s
    )


# --------------------------------------------------------------------------- #
# Curve arithmetic
# --------------------------------------------------------------------------- #
def test_retention_weights_by_overlap():
    """A window straddling two buckets is credited proportionally."""
    hm = _heatmap([0.0, 1.0])
    # 5s of the 0.0 bucket + 5s of the 1.0 bucket = 0.5.
    assert hm.retention_for(5.0, 15.0) == pytest.approx(0.5)


def test_lift_is_relative_to_video_mean():
    hm = _heatmap([0.2, 0.2, 0.2, 0.6])
    assert hm.mean_value == pytest.approx(0.3)
    assert hm.lift_for(30.0, 40.0) == pytest.approx(2.0)


def test_empty_window_has_no_retention():
    hm = _heatmap([0.5, 0.5])
    assert hm.retention_for(10.0, 10.0) == 0.0


# --------------------------------------------------------------------------- #
# Peak detection
# --------------------------------------------------------------------------- #
def test_finds_the_spike():
    values = [0.2] * 10
    values[5] = values[6] = 0.95  # 50-70s is heavily replayed
    peaks = _heatmap(values).peaks(window_s=20.0)

    assert len(peaks) >= 1
    assert peaks[0].start_s == pytest.approx(50.0)
    assert peaks[0].lift > 1.15


def test_flat_curve_yields_no_peaks():
    """Nothing is 'most replayed' when everything is watched equally."""
    assert _heatmap([0.5] * 10).peaks() == []


def test_peaks_do_not_overlap():
    values = [0.1] * 20
    values[3] = values[4] = 0.9
    values[12] = values[13] = 0.9
    peaks = _heatmap(values).peaks(max_peaks=5, window_s=20.0)

    for a, b in zip(peaks, peaks[1:]):
        assert a.end_s <= b.start_s


def test_peaks_stay_inside_the_video():
    values = [0.1] * 10
    values[-1] = 0.9  # spike at the very end
    hm = _heatmap(values)
    for p in hm.peaks(window_s=30.0):
        assert p.start_s >= 0.0
        assert p.end_s <= 100.0


def test_short_video_keeps_fine_resolution():
    """A 3-minute video gets ~1.8s buckets, so 30s windows stay precise."""
    hm = _heatmap([0.5] * 100, bucket_s=1.8)
    assert not hm.is_coarse
    assert hm.bucket_width_s == pytest.approx(1.8)


def test_long_video_widens_the_peak_window():
    """YouTube always returns 100 buckets, so a 2h video gets ~74s buckets.

    A 30s window would sit inside one bucket and collapse onto its leading
    edge; the window must widen to cover the whole hot region instead.
    """
    values = [0.1] * 100
    values[50] = 0.9
    hm = _heatmap(values, bucket_s=73.9)  # ~2h video

    assert hm.is_coarse
    peaks = hm.peaks(max_peaks=3, window_s=30.0)
    assert peaks
    # Window spans the bucket rather than the requested 30s...
    assert peaks[0].duration_s == pytest.approx(73.9, abs=0.1)
    # ...but never exceeds the pipeline's maximum clip length.
    assert peaks[0].duration_s <= MAX_PEAK_WINDOW_S


def test_window_never_exceeds_max_clip_length():
    """Even absurdly wide buckets stay clippable."""
    values = [0.1] * 100
    values[50] = 0.9
    hm = _heatmap(values, bucket_s=600.0)  # ~16h video

    for p in hm.peaks(window_s=30.0):
        assert p.duration_s <= MAX_PEAK_WINDOW_S


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def _candidate(clip_id="clip_1", start=0.0, end=30.0, score=8, lift=None):
    return ClipCandidate(
        clip_id=clip_id,
        start_s=start,
        end_s=end,
        transcript_text="text",
        reason_chosen="because",
        scores=ClipScores(hook_potential=score, completeness=score, shareability=score),
        retention_lift=lift,
    )


def test_ranked_score_falls_back_to_llm_score_without_heatmap():
    c = _candidate(score=7)
    assert c.retention_lift is None
    assert c.ranked_score == float(c.total_score) == 21.0


def test_high_retention_outranks_a_better_llm_score():
    """The whole point: a rewatched clip beats a 'better written' dead one."""
    replayed = _candidate("clip_1", score=8, lift=1.6)
    dead = _candidate("clip_2", score=9, lift=0.6)

    assert dead.total_score > replayed.total_score
    assert replayed.ranked_score > dead.ranked_score


def test_retention_bonus_is_capped():
    """View data nudges ranking; it can't swamp the LLM's judgement."""
    extreme = _candidate(score=5, lift=99.0)
    assert extreme.ranked_score == extreme.total_score + 4.0


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def test_no_heatmap_leaves_candidates_untouched():
    cands = [_candidate()]
    assert _fuse_heatmap(cands, None, _transcript(), 6) == cands


def test_unfound_peak_becomes_a_candidate():
    values = [0.1] * 10
    values[5] = values[6] = 0.95
    hm = _heatmap(values)
    # The LLM only looked at the start of the video.
    cands = [_candidate("clip_1", start=0.0, end=20.0)]

    out = _fuse_heatmap(cands, hm, _transcript(), 6)
    added = [c for c in out if c.origin == "heatmap"]

    assert len(added) == 1
    assert added[0].start_s >= 40.0
    assert added[0].retention_lift > 1.15
    assert "replayed" in added[0].reason_chosen


def test_peak_already_covered_is_not_duplicated():
    values = [0.1] * 10
    values[5] = values[6] = 0.95
    hm = _heatmap(values)
    # An LLM clip spanning the whole spike.
    cands = [_candidate("clip_1", start=40.0, end=80.0)]

    out = _fuse_heatmap(cands, hm, _transcript(), 6)

    assert [c for c in out if c.origin == "heatmap"] == []
    assert out[0].retention_lift is not None  # still annotated


def test_peak_without_transcript_text_is_skipped():
    """Textless peaks are dropped — downstream agents write from text."""
    values = [0.1] * 10
    values[5] = values[6] = 0.95
    hm = _heatmap(values)
    silent = Transcript(
        video_id="vid",
        language="en",
        segments=[Segment(start_s=0.0, end_s=10.0, text="only talking here")],
        total_duration_s=100.0,
    )

    out = _fuse_heatmap([], hm, silent, 6)

    assert out == []


def test_existing_candidates_get_annotated():
    hm = _heatmap([0.2, 0.2, 0.8, 0.8, 0.2])
    cands = [_candidate("clip_1", start=20.0, end=40.0)]

    out = _fuse_heatmap(cands, hm, _transcript(50), 6)

    assert out[0].retention_lift == pytest.approx(1.818, abs=1e-2)


def test_text_between_joins_overlapping_segments_only():
    t = _transcript(50)
    assert _text_between(t, 20.0, 40.0) == "words at 20 words at 30"


def test_lift_maps_into_valid_score_range():
    for lift in (1.15, 1.4, 2.0, 10.0):
        s = _lift_to_scores(lift)
        assert 1 <= s.hook_potential <= 10
        assert 1 <= s.shareability <= 10
