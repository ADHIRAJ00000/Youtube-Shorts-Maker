"""Coordinator / Ingestion agent.

Ensures the state has a transcript + content map (fetching/chunking if the
caller didn't pre-load them), then classifies the content type which drives
routing — short-form videos skip the clip-selection/critique pipeline.
"""

from __future__ import annotations

from app.agents.state import ContentState
from app.ingestion.chunker import build_content_map
from app.ingestion.transcript import load_transcript
from app.observability.logging_setup import get_logger

log = get_logger("app.agents.coordinator")

# Videos shorter than this are treated as already short-form.
SHORT_FORM_MAX_S = 180.0


def coordinator_node(state: ContentState) -> dict:
    transcript = state.get("transcript")
    content_map = state.get("content_map")

    # Self-contained path: ingest from the URL/file if not pre-loaded.
    if transcript is None:
        source = state.get("video_url")
        if not source:
            return {
                "errors": ["coordinator: no transcript and no video_url provided"],
                "node_trace": ["coordinator"],
            }
        transcript = load_transcript(source)  # raises TranscriptError -> resilient wrapper

    if content_map is None:
        content_map = build_content_map(transcript)

    duration = transcript.total_duration_s
    content_type = "short_form" if 0.0 < duration < SHORT_FORM_MAX_S else (
        state.get("content_type") or "unknown"
    )

    log.info(
        "coordinator",
        extra={"extra_fields": {
            "video_id": transcript.video_id,
            "duration_s": duration,
            "blocks": len(content_map.blocks),
            "content_type": content_type,
        }},
    )
    return {
        "transcript": transcript,
        "content_map": content_map,
        "content_type": content_type,
        "critic_round": 0,
        "node_trace": ["coordinator"],
    }
