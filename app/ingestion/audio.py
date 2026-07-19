"""Audio-transcription fallback via Groq Whisper.

When a video has no captions, this downloads its audio track and transcribes it
with Groq's `whisper-large-v3-turbo`, producing the same timestamped
`Transcript` the caption path produces. Downstream agents can't tell the
difference, so hooks/titles/SEO all work on caption-less videos.

This is the *fallback*, not the default: captions are free and instant, whereas
this costs an audio download plus ~$0.001 of transcription. It also still needs
YouTube access, so it does not work around a hosted-IP block — uploading a
transcript remains the escape hatch there.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from app.ingestion.transcript import Segment, Transcript, TranscriptError, extract_video_id
from app.observability.logging_setup import get_logger

log = get_logger("app.ingestion.audio")

# Groq's audio endpoint caps uploads (25 MB on the free tier). 16 kHz mono is
# what Whisper consumes internally anyway, so downsampling loses no accuracy
# and keeps roughly 2h of audio under the cap.
_TARGET_SAMPLE_RATE = "16000"
_MAX_UPLOAD_BYTES = 24 * 1024 * 1024

# Anything longer than ~50 min exceeds the upload cap at 64 kbps, so long
# videos are split and transcribed piece by piece. 20-minute pieces are ~9.6 MB
# — comfortably under the cap with headroom for variable bitrate.
_CHUNK_SECONDS = 20 * 60

WHISPER_MODEL = "whisper-large-v3-turbo"

# Whisper hallucinates on non-speech audio: fed music or silence it emits
# training-data artifacts ("Thank you.", "Suscríbete al canal!") scattered
# across the timeline. Real speech runs ~2-3 words/sec; hallucinated output
# lands near 0.2. Below this floor we reject the transcript rather than let
# the agents write hooks from invented dialogue.
_MIN_WORDS_PER_SECOND = 0.5

# Groq audio pricing (USD per hour of audio) — nominal, for the cost metric.
WHISPER_COST_PER_HOUR = 0.04


def _download_audio(video_url: str, dest_dir: Path) -> Path:
    """Download a video's audio track, downsampled to 16 kHz mono MP3."""
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise TranscriptError("yt-dlp is required for audio transcription.") from exc

    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "audio.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ],
        # Mono @ 16 kHz keeps the upload under Groq's size cap.
        "postprocessor_args": {"extractaudio": ["-ac", "1", "-ar", _TARGET_SAMPLE_RATE]},
    }

    from app.ingestion.heatmap import _proxy_url

    proxy = _proxy_url()
    if proxy:
        opts["proxy"] = proxy

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(video_url, download=True)
    except Exception as exc:
        raise TranscriptError(
            f"Could not download audio for transcription: {exc}"
        ) from exc

    files = sorted(dest_dir.glob("audio.*"))
    if not files:
        raise TranscriptError(
            "Audio download produced no file. ffmpeg may not be installed — "
            "it is required to transcribe videos without captions."
        )
    return files[0]


def _probe_duration(path: Path) -> float:
    """Exact duration of an audio file via ffprobe, 0.0 if unavailable."""
    import subprocess

    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60, check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def _split_audio(path: Path, chunk_s: int = _CHUNK_SECONDS) -> list[tuple[Path, float]]:
    """Split audio into chunks, returning (path, start_offset_s) pairs.

    Offsets come from probing each chunk's real duration rather than assuming
    `chunk_s` exactly: ffmpeg's stream-copy segmenter cuts on frame boundaries,
    so chunks drift by a fraction of a second each. Accumulating measured
    durations keeps timestamps aligned across a two-hour video instead of
    letting that drift compound into a multi-second offset by the end.
    """
    import subprocess

    out_dir = path.parent / "chunks"
    out_dir.mkdir(exist_ok=True)
    pattern = str(out_dir / "part%03d.mp3")

    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "segment",
             "-segment_time", str(chunk_s), "-c", "copy", pattern],
            capture_output=True, timeout=600, check=True,
        )
    except Exception as exc:
        raise TranscriptError(f"Could not split audio for transcription: {exc}") from exc

    chunks = sorted(out_dir.glob("part*.mp3"))
    if not chunks:
        raise TranscriptError("Audio splitting produced no chunks.")

    paired: list[tuple[Path, float]] = []
    offset = 0.0
    for c in chunks:
        paired.append((c, offset))
        offset += _probe_duration(c) or chunk_s
    return paired


def _transcribe_file(client, path: Path):
    """Send one audio file to Groq Whisper."""
    with open(path, "rb") as fh:
        return client.audio.transcriptions.create(
            file=(path.name, fh.read()),
            model=WHISPER_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )


def _groq_client():
    from app.config import get_settings

    settings = get_settings()
    if settings.llm_provider != "groq":
        raise TranscriptError(
            "Audio transcription requires a Groq API key (LLM_PROVIDER=groq)."
        )
    try:
        from groq import Groq  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise TranscriptError("The `groq` package is required for transcription.") from exc

    return Groq(api_key=settings.llm_api_key)


def transcribe_from_audio(video_url: str) -> Transcript:
    """Download a YouTube video's audio and transcribe it with Groq Whisper.

    Raises `TranscriptError` on any failure so callers treat it exactly like a
    failed caption fetch.
    """
    video_id = extract_video_id(video_url)
    log.info("audio.transcribe_start", extra={"extra_fields": {"video_id": video_id}})

    with tempfile.TemporaryDirectory(prefix="yt-audio-") as tmp:
        path = _download_audio(video_url, Path(tmp))
        client = _groq_client()

        # Long videos exceed the upload cap and must be transcribed in pieces,
        # each chunk's timestamps shifted back into whole-video time.
        if path.stat().st_size > _MAX_UPLOAD_BYTES:
            pieces = _split_audio(path)
            log.info(
                "audio.split",
                extra={"extra_fields": {"video_id": video_id, "chunks": len(pieces)}},
            )
        else:
            pieces = [(path, 0.0)]

        segments: list[Segment] = []
        total_duration = 0.0
        for i, (chunk_path, offset) in enumerate(pieces, start=1):
            try:
                result = _transcribe_file(client, chunk_path)
            except Exception as exc:
                raise TranscriptError(
                    f"Groq transcription failed on part {i}/{len(pieces)}: {exc}"
                ) from exc

            for seg in _segments_from_result(result):
                segments.append(
                    Segment(
                        start_s=seg.start_s + offset,
                        end_s=seg.end_s + offset,
                        text=seg.text,
                    )
                )
            chunk_dur = float(getattr(result, "duration", 0.0) or 0.0)
            total_duration = max(total_duration, offset + chunk_dur)
            if len(pieces) > 1:
                log.info(
                    "audio.chunk_done",
                    extra={"extra_fields": {"chunk": i, "of": len(pieces),
                                            "segments": len(segments)}},
                )

        language = str(getattr(result, "language", "") or "unknown")
    if not segments:
        raise TranscriptError(
            f"Transcription of {video_id} produced no speech — the video may have no dialogue."
        )

    # `total_duration` accumulates across chunks; `result` holds only the last.
    duration = total_duration or segments[-1].end_s
    words = sum(len(s.text.split()) for s in segments)
    density = words / duration if duration else 0.0
    if density < _MIN_WORDS_PER_SECOND:
        log.warning(
            "audio.transcript_rejected_low_speech",
            extra={"extra_fields": {"video_id": video_id, "words_per_second": round(density, 3)}},
        )
        raise TranscriptError(
            f"Video {video_id} has captions disabled and its audio contains little or no "
            "speech (music, ambience, or visual-only content), so it cannot be transcribed "
            "reliably. Clips still work if you upload a transcript file."
        )

    transcript = Transcript(
        video_id=video_id,
        language=language,
        segments=segments,
        total_duration_s=float(duration),
        source="whisper",
    )
    log.info(
        "audio.transcribe_done",
        extra={"extra_fields": {
            "video_id": video_id,
            "segments": len(segments),
            "duration_s": round(float(duration), 1),
            "cost_usd": round(float(duration) / 3600 * WHISPER_COST_PER_HOUR, 6),
        }},
    )
    return transcript


def _segments_from_result(result) -> list[Segment]:
    """Normalize Groq's verbose_json segments into our `Segment` model."""
    raw = getattr(result, "segments", None)
    if raw is None and isinstance(result, dict):
        raw = result.get("segments")

    segments: list[Segment] = []
    for item in raw or []:
        get = item.get if isinstance(item, dict) else lambda k, d=None: getattr(item, k, d)
        text = (get("text") or "").strip()
        if not text:
            continue
        try:
            segments.append(
                Segment(
                    start_s=float(get("start", 0.0)),
                    end_s=float(get("end", 0.0)),
                    text=text,
                )
            )
        except (TypeError, ValueError):
            continue
    return segments


def whisper_available() -> bool:
    """True if audio transcription is plausibly configured (Groq key + ffmpeg)."""
    from shutil import which

    from app.config import get_settings

    try:
        settings = get_settings()
    except Exception:
        return False
    return settings.llm_provider == "groq" and which("ffmpeg") is not None
