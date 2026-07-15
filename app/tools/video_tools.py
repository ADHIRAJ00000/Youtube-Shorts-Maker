"""Video tools + timestamp utilities.

Pure-python timestamp helpers (used everywhere) plus LangChain `@tool`
wrappers around transcript fetching and optional yt-dlp metadata, each
returning a uniform `{"ok": bool, "data": ..., "error": ...}` envelope so an
agent can reason about failure.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.tools import tool
from pydantic import BaseModel

from app.ingestion.transcript import TranscriptError, fetch_transcript
from app.observability.logging_setup import get_logger

log = get_logger("app.tools.video_tools")


# --------------------------------------------------------------------------- #
# Timestamp utilities (pure, deterministic)
# --------------------------------------------------------------------------- #
def seconds_to_hhmmss(seconds: float) -> str:
    """Format seconds as `hh:mm:ss` (or `mm:ss` if under an hour)."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def clip_duration(start_s: float, end_s: float) -> float:
    """Duration of a clip, clamped at 0."""
    return round(max(0.0, end_s - start_s), 3)


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """True if the two [start, end] intervals overlap at all."""
    return a_start < b_end and b_start < a_end


def overlap_fraction(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> float:
    """Fraction of the *shorter* interval that overlaps the other (0..1).

    Used by evals to decide whether an agent clip 'hits' a golden clip.
    """
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    shorter = min(a_end - a_start, b_end - b_start)
    if shorter <= 0:
        return 0.0
    return round(inter / shorter, 4)


def clamp_to_bounds(
    start_s: float, end_s: float, lower: float, upper: float
) -> tuple[float, float]:
    """Clamp a [start, end] range into [lower, upper], preserving order."""
    start = min(max(start_s, lower), upper)
    end = min(max(end_s, lower), upper)
    if end < start:
        start, end = end, start
    return round(start, 3), round(end, 3)


# --------------------------------------------------------------------------- #
# Tool return envelope
# --------------------------------------------------------------------------- #
class ToolResult(TypedDict):
    ok: bool
    data: Any
    error: str | None


def _ok(data: Any) -> ToolResult:
    return {"ok": True, "data": data, "error": None}


def _err(msg: str) -> ToolResult:
    return {"ok": False, "data": None, "error": msg}


class VideoMeta(BaseModel):
    """Lightweight video metadata (best-effort, may be partial)."""

    video_id: str
    title: str | None = None
    duration_s: float | None = None
    channel: str | None = None
    uploader: str | None = None


# --------------------------------------------------------------------------- #
# LangChain tools
# --------------------------------------------------------------------------- #
@tool
def fetch_transcript_tool(video_url: str) -> ToolResult:
    """Fetch the timestamped transcript for a YouTube video URL or id.

    Returns {"ok", "data", "error"}; on success `data` is a serialized
    Transcript (video_id, language, segments, total_duration_s).
    """
    try:
        transcript = fetch_transcript(video_url)
        return _ok(transcript.model_dump())
    except TranscriptError as exc:
        return _err(str(exc))
    except Exception as exc:  # defensive
        return _err(f"Unexpected transcript error: {exc}")


@tool
def fetch_video_metadata_tool(video_url: str) -> ToolResult:
    """Fetch best-effort video metadata (title, duration, channel) via yt-dlp.

    Optional context for the agents. Returns {"ok","data","error"}; failures
    are non-fatal — the pipeline works without metadata.
    """
    return fetch_video_metadata(video_url)


def fetch_video_metadata(video_url: str) -> ToolResult:
    """Non-tool callable version of the metadata fetch (used internally)."""
    try:
        from yt_dlp import YoutubeDL  # type: ignore

        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": False,
        }
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
        meta = VideoMeta(
            video_id=info.get("id", ""),
            title=info.get("title"),
            duration_s=float(info["duration"]) if info.get("duration") else None,
            channel=info.get("channel") or info.get("uploader"),
            uploader=info.get("uploader"),
        )
        return _ok(meta.model_dump())
    except Exception as exc:
        # Metadata is optional; never fatal.
        log.warning(
            "metadata.fetch_failed",
            extra={"extra_fields": {"error": str(exc)}},
        )
        return _err(f"Could not fetch metadata: {exc}")


VIDEO_TOOLS = [fetch_transcript_tool, fetch_video_metadata_tool]
