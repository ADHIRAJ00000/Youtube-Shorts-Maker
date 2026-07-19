"""Pipeline orchestration shared by the API and scripts.

Ingestion is done up front (so a bad URL fails fast with a clean error) and the
resulting transcript + content map are handed to the graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.agents.graph import build_graph
from app.agents.state import ContentState
from app.config import get_settings
from app.ingestion.audio import transcribe_from_audio, whisper_available
from app.ingestion.chunker import build_content_map
from app.ingestion.heatmap import fetch_heatmap
from app.ingestion.transcript import TranscriptError, load_transcript
from app.observability import tracing
from app.observability.logging_setup import get_logger

log = get_logger("app.pipeline")


def build_initial_state(source: str, video_url: Optional[str] = None) -> ContentState:
    """Ingest a URL/file into an initial graph state. Raises TranscriptError."""
    is_url = not Path(source).exists()
    errors: list[str] = []

    try:
        transcript = load_transcript(source)
    except TranscriptError:
        # No captions. For a URL we can still transcribe the audio ourselves;
        # for an uploaded file there is nothing else to try.
        if not (is_url and whisper_available()):
            raise
        log.info("ingest.captions_missing_falling_back_to_whisper")
        transcript = transcribe_from_audio(source)
        errors.append("No captions available; transcribed the audio with Whisper.")

    content_map = build_content_map(transcript)

    # Audience retention is optional enrichment — never fatal, only for URLs.
    heatmap = fetch_heatmap(source) if is_url else None

    return {
        "video_url": video_url if video_url is not None else source,
        "transcript": transcript,
        "content_map": content_map,
        "heatmap": heatmap,
        "content_type": "unknown",
        "node_trace": [],
        "errors": errors,
        "token_usage": {},
    }


def run_pipeline(state: ContentState, job_id: Optional[str] = None) -> ContentState:
    """Run the full agent graph on a prepared initial state.

    When Langfuse is configured, one trace per job is created with a span per
    agent invocation (the critique loop appears as repeated spans).
    """
    settings = get_settings()
    graph = build_graph(max_critic_rounds=settings.max_critic_rounds)
    config = tracing.trace_config(session_id=job_id)
    try:
        return graph.invoke(state, config=config) if config else graph.invoke(state)
    finally:
        tracing.flush(config)


def job_metrics(final: ContentState) -> dict[str, Any]:
    """Extract the numbers the job store records from a finished state."""
    pkg = final.get("final_package") or {}
    critique = pkg.get("critique", {})
    cost = pkg.get("cost", {})
    outputs = pkg.get("outputs", {})
    return {
        "clips_proposed": critique.get("proposed", 0),
        "clips_approved": critique.get("approved", 0),
        "clips_rejected": critique.get("rejected", 0),
        "critic_rounds": critique.get("rounds", 0),
        "total_tokens": cost.get("total_tokens", 0),
        "cost_usd": cost.get("total_cost_usd", 0.0),
        "package_path": outputs.get("markdown", ""),
        "package": pkg,
    }
