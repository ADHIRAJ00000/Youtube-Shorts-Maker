"""Clip selection / approval gate.

Runs once the critique loop decides to proceed. Turns candidates + the latest
critic verdicts into a ranked list of `ApprovedClip`s. If the critic rejected
everything (e.g. after max rounds), it falls back to the top-2 by score and
records a warning in `errors` — the pipeline must always complete.
"""

from __future__ import annotations

from app.agents.state import ApprovedClip, ClipCandidate, ContentState, CriticNote
from app.observability.logging_setup import get_logger

log = get_logger("app.agents.select_clips")

_FALLBACK_N = 2


def select_clips_node(state: ContentState) -> dict:
    candidates: list[ClipCandidate] = state.get("clip_candidates", []) or []
    feedback: list[CriticNote] = state.get("critic_feedback", []) or []
    verdict_by_id = {n.clip_id: n for n in feedback}

    # Approved = explicitly approved, or (defensively) unmentioned by the critic.
    approved = [
        c
        for c in candidates
        if verdict_by_id.get(c.clip_id) is None
        or verdict_by_id[c.clip_id].verdict == "approve"
    ]

    errors: list[str] = []
    if not approved and candidates:
        approved = sorted(candidates, key=lambda c: c.ranked_score, reverse=True)[:_FALLBACK_N]
        errors.append(
            f"All clips rejected after critique; fell back to top-{_FALLBACK_N} by score."
        )
        log.warning("select_clips.fallback", extra={"extra_fields": {"kept": len(approved)}})

    ranked = sorted(approved, key=lambda c: c.ranked_score, reverse=True)
    approved_clips = [
        ApprovedClip(
            candidate=c,
            verdict=(verdict_by_id[c.clip_id].verdict if c.clip_id in verdict_by_id else "approve"),
            rank=i + 1,
        )
        for i, c in enumerate(ranked)
    ]

    rejected = len(candidates) - len(approved_clips)
    log.info(
        "select_clips.done",
        extra={"extra_fields": {"proposed": len(candidates), "approved": len(approved_clips), "rejected": rejected}},
    )

    out: dict = {"approved_clips": approved_clips, "node_trace": ["select_clips"]}
    if errors:
        out["errors"] = errors
    return out
