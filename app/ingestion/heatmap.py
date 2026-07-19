"""YouTube 'most replayed' heatmap ingestion.

YouTube exposes a 100-bucket retention curve for many videos — the wavy graph
drawn over the scrubber. Peaks are the moments viewers rewatch, which is a
*behavioural* signal about what is interesting, entirely independent of what
the transcript says.

Two uses here:
  * rank/boost transcript-derived clips that land on a peak, and
  * seed clip candidates directly from peaks (the only signal available when a
    video has no captions at all).

The heatmap is best-effort: not every video has one, and fetching needs
YouTube access. Every failure returns None rather than raising — a missing
heatmap must never fail a job.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.observability.logging_setup import get_logger

log = get_logger("app.ingestion.heatmap")

# Window length used when scanning the curve for peak moments.
DEFAULT_PEAK_WINDOW_S = 30.0

# YouTube always returns exactly 100 buckets, so bucket width scales with the
# video: ~1.8s for a 3-minute clip but ~74s for a 2-hour podcast. Once buckets
# are wider than the scan window, a 30s window sits entirely inside one bucket
# and every peak collapses onto a bucket's leading edge — the real moment could
# be anywhere in the remaining ~44s. So the window grows to cover the whole hot
# bucket, capped at the pipeline's maximum clip length.
MAX_PEAK_WINDOW_S = 75.0

# A peak must beat the video's average retention by this factor to count as a
# real "most replayed" moment rather than curve noise.
PEAK_THRESHOLD_RATIO = 1.15


class HeatmapBucket(BaseModel):
    """One sample of the retention curve."""

    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    value: float = Field(ge=0)


class HeatmapPeak(BaseModel):
    """A high-retention window found by scanning the curve."""

    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    score: float = Field(ge=0, description="Mean retention across the window (0-1).")
    lift: float = Field(ge=0, description="Score relative to the video mean (1.0 = average).")

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)


class Heatmap(BaseModel):
    """The full retention curve for one video."""

    video_id: str
    buckets: list[HeatmapBucket]

    @property
    def bucket_width_s(self) -> float:
        """Width of one bucket — the curve's time resolution."""
        if not self.buckets:
            return 0.0
        return self.buckets[0].end_s - self.buckets[0].start_s

    @property
    def is_coarse(self) -> bool:
        """True when buckets are too wide to locate a moment precisely.

        Signals that peak timings are approximate (long video), so the clip
        boundaries are best treated as a region to trim rather than exact cuts.
        """
        return self.bucket_width_s > DEFAULT_PEAK_WINDOW_S

    @property
    def mean_value(self) -> float:
        if not self.buckets:
            return 0.0
        return sum(b.value for b in self.buckets) / len(self.buckets)

    def retention_for(self, start_s: float, end_s: float) -> float:
        """Mean retention across an arbitrary window, 0.0 if it covers nothing.

        Buckets are weighted by how much of them the window actually overlaps,
        so a clip that clips a bucket's edge isn't credited for all of it.
        """
        if end_s <= start_s:
            return 0.0
        total_weight = 0.0
        weighted = 0.0
        for b in self.buckets:
            overlap = min(end_s, b.end_s) - max(start_s, b.start_s)
            if overlap > 0:
                weighted += b.value * overlap
                total_weight += overlap
        return round(weighted / total_weight, 4) if total_weight else 0.0

    def lift_for(self, start_s: float, end_s: float) -> float:
        """Retention of a window relative to the video's mean (1.0 = average)."""
        mean = self.mean_value
        if mean <= 0:
            return 0.0
        return round(self.retention_for(start_s, end_s) / mean, 4)

    def peaks(
        self,
        max_peaks: int = 5,
        window_s: float = DEFAULT_PEAK_WINDOW_S,
        threshold_ratio: float = PEAK_THRESHOLD_RATIO,
    ) -> list[HeatmapPeak]:
        """Find the top non-overlapping high-retention windows.

        Slides a `window_s` window across the curve at bucket resolution,
        scores each position by mean retention, then greedily takes the best
        positions that don't overlap one already taken. Only windows beating
        the video mean by `threshold_ratio` qualify, so a flat curve (nothing
        is especially replayed) correctly yields no peaks.
        """
        if not self.buckets:
            return []

        mean = self.mean_value
        if mean <= 0:
            return []

        video_end = self.buckets[-1].end_s

        # On long videos widen the window to span a whole bucket, so the clip
        # covers the hot region rather than clipping its leading edge.
        bucket_s = self.bucket_width_s
        if bucket_s > window_s:
            window_s = min(bucket_s, MAX_PEAK_WINDOW_S)

        window_s = min(window_s, video_end)
        if window_s <= 0:
            return []

        # Score a window starting at each bucket boundary.
        scored: list[HeatmapPeak] = []
        for b in self.buckets:
            start = b.start_s
            end = start + window_s
            if end > video_end:
                # Anchor the final window to the end rather than running past it.
                start, end = max(0.0, video_end - window_s), video_end
            score = self.retention_for(start, end)
            if score / mean >= threshold_ratio:
                scored.append(
                    HeatmapPeak(
                        start_s=round(start, 3),
                        end_s=round(end, 3),
                        score=score,
                        lift=round(score / mean, 4),
                    )
                )

        # Greedy non-overlapping selection, best first.
        scored.sort(key=lambda p: p.score, reverse=True)
        chosen: list[HeatmapPeak] = []
        for cand in scored:
            if all(cand.start_s >= c.end_s or cand.end_s <= c.start_s for c in chosen):
                chosen.append(cand)
            if len(chosen) >= max_peaks:
                break

        chosen.sort(key=lambda p: p.start_s)
        return chosen


def fetch_heatmap(video_url: str) -> Heatmap | None:
    """Fetch a video's retention curve via yt-dlp. Returns None if unavailable.

    Uses `process=False` so yt-dlp only parses the player response and skips
    format selection — the heatmap lives in the metadata, and format selection
    fails on videos with restricted streams even when metadata is fine.
    """
    try:
        import yt_dlp  # type: ignore

        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "noplaylist": True,
        }
        proxy = _proxy_url()
        if proxy:
            opts["proxy"] = proxy

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(video_url, download=False, process=False)
    except Exception as exc:
        log.warning("heatmap.fetch_failed", extra={"extra_fields": {"error": str(exc)}})
        return None

    raw = (info or {}).get("heatmap")
    if not raw:
        log.info("heatmap.unavailable", extra={"extra_fields": {"video": video_url}})
        return None

    buckets: list[HeatmapBucket] = []
    for item in raw:
        try:
            buckets.append(
                HeatmapBucket(
                    start_s=float(item["start_time"]),
                    end_s=float(item["end_time"]),
                    value=float(item["value"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    if not buckets:
        return None

    hm = Heatmap(video_id=str((info or {}).get("id", "")), buckets=buckets)
    log.info(
        "heatmap.fetched",
        extra={"extra_fields": {"buckets": len(buckets), "mean": round(hm.mean_value, 4)}},
    )
    return hm


def _proxy_url() -> str | None:
    """Reuse the same proxy env vars the transcript fetcher honours."""
    import os

    user = os.environ.get("YOUTUBE_PROXY_USERNAME", "").strip()
    pw = os.environ.get("YOUTUBE_PROXY_PASSWORD", "").strip()
    if user and pw:
        return f"http://{user}:{pw}@p.webshare.io:80"
    return (
        os.environ.get("YOUTUBE_HTTPS_PROXY", "").strip()
        or os.environ.get("YOUTUBE_HTTP_PROXY", "").strip()
        or None
    )
