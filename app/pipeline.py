"""Pipeline orchestration shared by the API and scripts.

Ingestion is done up front (so a bad URL fails fast with a clean error) and the
resulting transcript + content map are handed to the graph.
"""

from __future__ import annotations

from typing import Any, Optional

from app.agents.graph import build_graph
from app.agents.state import ContentState
from app.config import get_settings
from app.ingestion.chunker import build_content_map
from app.ingestion.transcript import load_transcript
from app.observability import tracing


def build_initial_state(source: str, video_url: Optional[str] = None) -> ContentState:
    """Ingest a URL/file into an initial graph state. Raises TranscriptError."""
    transcript = load_transcript(source)
    content_map = build_content_map(transcript)
    return {
        "video_url": video_url if video_url is not None else source,
        "transcript": transcript,
        "content_map": content_map,
        "content_type": "unknown",
        "node_trace": [],
        "errors": [],
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
