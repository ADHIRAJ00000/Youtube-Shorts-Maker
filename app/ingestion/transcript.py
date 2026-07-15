"""Transcript fetching and normalization.

Turns a YouTube URL (via `youtube-transcript-api`) or a local transcript file
(`.txt` / `.srt` / `.json`) into a typed, timestamped `Transcript` model.

No LLM is involved here — this is pure, deterministic ingestion.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field, model_validator

from app.observability.logging_setup import get_logger

log = get_logger("app.ingestion.transcript")

# Assumed speaking rate used only when a plain-text file has no timestamps.
_WORDS_PER_SECOND = 2.5

# Language preference order when a video offers several transcripts.
DEFAULT_LANGUAGE_PREFERENCE: tuple[str, ...] = ("en", "en-US", "en-GB", "hi")


class TranscriptError(Exception):
    """Raised for any recoverable ingestion failure (bad URL, no transcript…).

    Callers should catch this and surface a clean typed error rather than a
    raw stack trace.
    """


def _build_proxy_config():
    """Build a youtube-transcript-api proxy config from env vars, or None.

    YouTube blocks datacenter IPs, so hosted deployments usually need a
    residential proxy. Supports Webshare (recommended) or a generic HTTP proxy.
    """
    import os

    user = os.environ.get("YOUTUBE_PROXY_USERNAME", "").strip()
    pw = os.environ.get("YOUTUBE_PROXY_PASSWORD", "").strip()
    if user and pw:
        from youtube_transcript_api.proxies import WebshareProxyConfig  # type: ignore

        return WebshareProxyConfig(proxy_username=user, proxy_password=pw)

    http = os.environ.get("YOUTUBE_HTTP_PROXY", "").strip()
    https = os.environ.get("YOUTUBE_HTTPS_PROXY", "").strip() or http
    if http or https:
        from youtube_transcript_api.proxies import GenericProxyConfig  # type: ignore

        return GenericProxyConfig(http_url=http or https, https_url=https or http)

    return None


class Segment(BaseModel):
    """A single timestamped line of transcript."""

    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _end_after_start(self) -> "Segment":
        if self.end_s < self.start_s:
            self.end_s = self.start_s
        self.text = self.text.strip()
        return self

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)


class Transcript(BaseModel):
    """A normalized, timestamped transcript for one video."""

    video_id: str
    language: str
    segments: list[Segment]
    total_duration_s: float = Field(ge=0)
    source: str = "youtube"  # "youtube" | "file" | "estimated"
    available_languages: list[str] = Field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.segments)

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)


# --------------------------------------------------------------------------- #
# URL / video-id helpers
# --------------------------------------------------------------------------- #
def extract_video_id(video_url: str) -> str:
    """Extract the 11-char video id from any common YouTube URL form.

    Also accepts a bare 11-char id. Raises TranscriptError on failure.
    """
    url = video_url.strip()

    # Bare id.
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")

    if host in {"youtu.be"}:
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/shorts/", "/embed/", "/v/", "/live/")):
            candidate = parsed.path.split("/")[2]
        else:
            candidate = ""
    else:
        candidate = ""

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        return candidate
    raise TranscriptError(f"Could not extract a YouTube video id from: {video_url!r}")


# --------------------------------------------------------------------------- #
# YouTube fetching
# --------------------------------------------------------------------------- #
def fetch_transcript(
    video_url: str,
    languages: Sequence[str] = DEFAULT_LANGUAGE_PREFERENCE,
) -> Transcript:
    """Fetch a transcript from YouTube for `video_url`.

    Prefers manually-created transcripts in the preferred languages, then
    falls back to auto-generated ones. Raises `TranscriptError` on any
    recoverable failure (no transcript, disabled, unavailable video).
    """
    # Imported lazily so unit tests that only touch file/chunker paths do not
    # require the network stack.
    from youtube_transcript_api import (  # type: ignore
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
        YouTubeTranscriptApi,
    )
    from youtube_transcript_api._errors import (  # type: ignore
        CouldNotRetrieveTranscript,
        IpBlocked,
        RequestBlocked,
    )

    video_id = extract_video_id(video_url)

    try:
        proxy_config = _build_proxy_config()
        api = YouTubeTranscriptApi(proxy_config=proxy_config) if proxy_config else YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        available = [t.language_code for t in transcript_list]

        chosen = None
        chosen_lang = None
        # 1) Prefer a manually-created transcript in a preferred language.
        try:
            picked = transcript_list.find_manually_created_transcript(list(languages))
            chosen, chosen_lang = picked, picked.language_code
        except Exception:
            pass
        # 2) Fall back to a generated transcript in a preferred language.
        if chosen is None:
            try:
                picked = transcript_list.find_generated_transcript(list(languages))
                chosen, chosen_lang = picked, picked.language_code
            except Exception:
                pass
        # 3) Last resort: the first available transcript of any language.
        if chosen is None and available:
            picked = next(iter(transcript_list))
            chosen, chosen_lang = picked, picked.language_code

        if chosen is None:
            raise TranscriptError(
                f"No transcript available for video {video_id}."
            )

        raw = chosen.fetch()
    except (IpBlocked, RequestBlocked) as exc:
        raise TranscriptError(
            "YouTube blocked this request. This is common on cloud/hosted servers "
            "(Render, AWS, etc.) whose IPs YouTube throttles. Options: (1) upload "
            "the transcript file (.srt/.txt/.json) instead — that path needs no "
            "YouTube access; or (2) set a residential proxy via the "
            "YOUTUBE_PROXY_USERNAME/YOUTUBE_PROXY_PASSWORD env vars."
        ) from exc
    except (TranscriptsDisabled,) as exc:
        raise TranscriptError(
            f"Transcripts are disabled for video {video_id}."
        ) from exc
    except (NoTranscriptFound,) as exc:
        raise TranscriptError(
            f"No transcript found for video {video_id} in languages {list(languages)}."
        ) from exc
    except (VideoUnavailable,) as exc:
        raise TranscriptError(f"Video {video_id} is unavailable.") from exc
    except CouldNotRetrieveTranscript as exc:
        raise TranscriptError(
            f"Could not retrieve transcript for {video_id}: {exc}"
        ) from exc
    except TranscriptError:
        raise
    except Exception as exc:  # network / parsing / API changes
        raise TranscriptError(
            f"Unexpected error fetching transcript for {video_id}: {exc}"
        ) from exc

    segments = _segments_from_raw(raw)
    if not segments:
        raise TranscriptError(f"Transcript for {video_id} was empty.")

    transcript = Transcript(
        video_id=video_id,
        language=chosen_lang or "unknown",
        segments=segments,
        total_duration_s=segments[-1].end_s,
        source="youtube",
        available_languages=available,
    )
    log.info(
        "transcript.fetched",
        extra={
            "extra_fields": {
                "video_id": video_id,
                "language": transcript.language,
                "segments": len(segments),
                "duration_s": transcript.total_duration_s,
            }
        },
    )
    return transcript


def _segments_from_raw(raw) -> list[Segment]:
    """Convert transcript rows to Segments.

    Accepts both plain dicts ({text,start,duration}, from JSON files or the
    youtube-transcript-api list form) and `FetchedTranscriptSnippet` objects
    (attributes text/start/duration, from a live 1.x `.fetch()`).
    """
    segments: list[Segment] = []
    for row in raw:
        if isinstance(row, dict):
            text = row.get("text") or ""
            start = float(row.get("start", 0.0))
            duration = float(row.get("duration", 0.0))
        else:  # FetchedTranscriptSnippet
            text = getattr(row, "text", "") or ""
            start = float(getattr(row, "start", 0.0))
            duration = float(getattr(row, "duration", 0.0))
        text = text.replace("\n", " ").strip()
        if not text:
            continue
        segments.append(Segment(start_s=start, end_s=start + duration, text=text))
    return segments


# --------------------------------------------------------------------------- #
# Local file loading (.txt / .srt / .json)
# --------------------------------------------------------------------------- #
def load_transcript_file(path: str | Path, video_id: str | None = None) -> Transcript:
    """Load a transcript from a local `.txt`, `.srt`, or `.json` file."""
    p = Path(path)
    if not p.exists():
        raise TranscriptError(f"Transcript file not found: {p}")

    vid = video_id or p.stem
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8", errors="replace")

    if suffix == ".srt":
        segments = _parse_srt(text)
        source = "file"
        language = "unknown"
    elif suffix == ".json":
        segments, language, source = _parse_json_transcript(text)
    elif suffix == ".txt":
        segments = _estimate_from_plaintext(text)
        source = "estimated"
        language = "unknown"
    else:
        raise TranscriptError(
            f"Unsupported transcript file type {suffix!r} (use .txt/.srt/.json)."
        )

    if not segments:
        raise TranscriptError(f"No usable segments parsed from {p}.")

    return Transcript(
        video_id=vid,
        language=language,
        segments=segments,
        total_duration_s=segments[-1].end_s,
        source=source,
    )


_SRT_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _srt_ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_srt(text: str) -> list[Segment]:
    """Parse SRT subtitle blocks into Segments."""
    segments: list[Segment] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        # Optional numeric index line, then the timing line.
        idx = 0
        if lines[0].strip().isdigit():
            idx = 1
        if idx >= len(lines):
            continue
        m = _SRT_TIME.search(lines[idx])
        if not m:
            continue
        start = _srt_ts_to_seconds(*m.groups()[:4])
        end = _srt_ts_to_seconds(*m.groups()[4:])
        content = " ".join(lines[idx + 1 :]).strip()
        if content:
            segments.append(Segment(start_s=start, end_s=end, text=content))
    return segments


def _parse_json_transcript(text: str) -> tuple[list[Segment], str, str]:
    """Parse JSON in either youtube-transcript-api list form or a Transcript dump."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TranscriptError(f"Invalid JSON transcript: {exc}") from exc

    # Our own Transcript dump.
    if isinstance(data, dict) and "segments" in data:
        segments = [
            Segment(
                start_s=float(s["start_s"]),
                end_s=float(s["end_s"]),
                text=str(s["text"]),
            )
            for s in data["segments"]
            if str(s.get("text", "")).strip()
        ]
        return segments, str(data.get("language", "unknown")), "file"

    # youtube-transcript-api list form: [{text,start,duration}, ...]
    if isinstance(data, list):
        return _segments_from_raw(data), "unknown", "file"

    raise TranscriptError("Unrecognized JSON transcript structure.")


def _estimate_from_plaintext(text: str) -> list[Segment]:
    """Build pseudo-timestamped segments from untimed plain text.

    Splits on sentence boundaries and assigns timestamps assuming a constant
    speaking rate. Marked as source='estimated' so downstream code knows the
    timestamps are approximate.
    """
    # Split into sentence-ish units, keeping the terminator.
    sentences = re.findall(r"[^.!?\n]+[.!?]?", text.replace("\r", " "))
    sentences = [s.strip() for s in sentences if s.strip()]

    segments: list[Segment] = []
    cursor = 0.0
    for sent in sentences:
        words = len(sent.split())
        dur = max(1.0, words / _WORDS_PER_SECOND)
        segments.append(Segment(start_s=cursor, end_s=cursor + dur, text=sent))
        cursor += dur
    return segments


def load_transcript(source: str) -> Transcript:
    """Dispatch: treat `source` as a local file if it exists, else a URL."""
    if Path(source).exists():
        return load_transcript_file(source)
    return fetch_transcript(source)
