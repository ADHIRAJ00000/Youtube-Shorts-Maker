"""Clip Scout agent.

Reads the timestamped content map and proposes the best short-form clips as
structured output. On revision rounds it receives its previous picks plus every
critic note and must address them. All timestamps are validated *in code*
against the transcript bounds — the prompt asks for honesty, the validator
enforces it.

Long videos use **map-reduce chunking**: the transcript is split into pieces
small enough for the free-tier per-request limit, the best clips are found in
each piece (the "map"), then the top ones overall are kept (the "reduce"). This
lets the agent handle videos up to ~1.5 hours without any single request
blowing the token cap.
"""

from __future__ import annotations

import math

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents import llm
from app.agents.state import (
    ClipCandidate,
    ClipPick,
    ClipPickList,
    ClipScores,
    ContentState,
    CriticNote,
)
from app.config import get_settings
from app.observability.logging_setup import get_logger
from app.tools.video_tools import clamp_to_bounds

log = get_logger("app.agents.clip_scout")

# Hard duration bounds (seconds). Picks outside this are clamped, not dropped.
MIN_CLIP_S = 15.0
MAX_CLIP_S = 75.0

# ~1.33 tokens per English word.
_TOKENS_PER_WORD = 1.33
# Above this many transcript tokens, find clips chunk-by-chunk. Kept low enough
# that a single request (transcript + system + schema hint + output reservation)
# stays under the 8B fallback's ~6k tokens/minute cap, so long videos never
# hard-fail even when the primary model is daily-capped.
CHUNK_THRESHOLD_TOKENS = 3000
# Target transcript tokens per chunk (~2500 transcript + ~950 overhead + 2000
# output reservation ≈ 5450, safely under the 6k TPM cap).
CHUNK_TOKENS = 2500

SYSTEM_PROMPT = """You are a short-form content strategist for a YouTube channel \
covering football and AI. You receive a timestamped content map of a video (it \
may be one section of a longer video). Identify the {max_clips} best segments \
for Shorts/Reels. A great clip is: self-contained (understandable without \
context), 20-60 seconds, opens strong or contains a surprising/emotional/\
high-value moment, and ends on a complete thought. For each pick, provide exact \
start/end timestamps FROM THE MAP, the transcript text, WHY you chose it, and \
scores (1-10) for hook_potential, completeness, and shareability. Do NOT pick \
segments that reference earlier parts of the video ("as I said before"). Do NOT \
invent timestamps - use only timestamps present in the content map. If you \
received critic feedback, address every point of it. Output ONLY the structured \
schema."""


# --------------------------------------------------------------------------- #
# Chunking helpers
# --------------------------------------------------------------------------- #
def _block_tokens(block) -> int:
    return int(block.word_count * _TOKENS_PER_WORD)


def _render_blocks(blocks) -> str:
    return "\n".join(
        f"[block {b.index} | {b.start_s:.1f}s-{b.end_s:.1f}s] {b.text}" for b in blocks
    )


def _chunk_blocks(blocks, max_tokens: int = CHUNK_TOKENS) -> list[list]:
    """Group content blocks into chunks that each stay under `max_tokens`."""
    chunks: list[list] = []
    cur: list = []
    cur_tokens = 0
    for b in blocks:
        bt = _block_tokens(b)
        if cur and cur_tokens + bt > max_tokens:
            chunks.append(cur)
            cur, cur_tokens = [], 0
        cur.append(b)
        cur_tokens += bt
    if cur:
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------- #
# Revision context (small-video single-call path)
# --------------------------------------------------------------------------- #
def _render_previous(state: ContentState) -> str:
    candidates = state.get("clip_candidates", []) or []
    feedback: list[CriticNote] = state.get("critic_feedback", []) or []
    if not candidates and not feedback:
        return ""
    lines = ["\n--- YOUR PREVIOUS PICKS ---"]
    for c in candidates:
        lines.append(
            f"[{c.clip_id}] {c.start_s:.1f}s-{c.end_s:.1f}s "
            f"(scores h{c.scores.hook_potential}/c{c.scores.completeness}/"
            f"s{c.scores.shareability}): {c.transcript_text[:120]}"
        )
    lines.append("\n--- CRITIC FEEDBACK YOU MUST ADDRESS ---")
    for n in feedback:
        sugg = f" | suggestion: {n.suggestions}" if n.suggestions else ""
        lines.append(f"[{n.clip_id}] {n.verdict.upper()}: {'; '.join(n.reasons)}{sugg}")
    lines.append(
        "\nProduce a REVISED set of picks that fixes every issue above. "
        "Drop clips that were rejected as fundamentally weak; fix clips marked "
        "for revision."
    )
    return "\n".join(lines)


def _revision_note(state: ContentState) -> str:
    """Compact critic feedback for the chunked-map revision path."""
    feedback: list[CriticNote] = state.get("critic_feedback", []) or []
    if not feedback:
        return ""
    issues = "; ".join(
        f"{n.verdict}: {' / '.join(n.reasons)}"
        for n in feedback
        if n.verdict in ("reject", "revise")
    )
    if not issues:
        return ""
    return (
        "\n\nNOTE: an earlier pass had these problems — avoid them and pick "
        f"stronger, self-contained moments: {issues}"
    )


# --------------------------------------------------------------------------- #
# LLM call + validation
# --------------------------------------------------------------------------- #
def _scout_call(rendered_map: str, max_clips: int, extra: str = "") -> tuple[list[ClipPick], dict]:
    system = SYSTEM_PROMPT.format(max_clips=max_clips)
    human = "CONTENT MAP (use only these timestamps):\n" + rendered_map + extra
    result, usage = llm.structured_invoke(
        ClipPickList,
        [SystemMessage(content=system), HumanMessage(content=human)],
        temperature=0.5,
    )
    return result.clips, usage


def _validate_clip(pick, total_duration_s: float) -> tuple[float, float, list[str]]:
    """Clamp a pick's timestamps to transcript bounds and the 15-75s window."""
    notes: list[str] = []
    start, end = clamp_to_bounds(pick.start_s, pick.end_s, 0.0, total_duration_s)
    if (start, end) != (round(pick.start_s, 3), round(pick.end_s, 3)):
        notes.append(f"clamped to transcript bounds [0,{total_duration_s:.0f}]")

    duration = end - start
    if duration < MIN_CLIP_S:
        end = min(start + MIN_CLIP_S, total_duration_s)
        if end - start < MIN_CLIP_S:
            start = max(0.0, end - MIN_CLIP_S)
        notes.append(f"duration {duration:.1f}s < {MIN_CLIP_S:.0f}s min; extended")
    elif duration > MAX_CLIP_S:
        end = start + MAX_CLIP_S
        notes.append(f"duration {duration:.1f}s > {MAX_CLIP_S:.0f}s max; trimmed")

    return round(start, 3), round(end, 3), notes


def _score(pick: ClipPick) -> int:
    return pick.hook_potential + pick.completeness + pick.shareability


# --------------------------------------------------------------------------- #
# Audience-retention fusion
# --------------------------------------------------------------------------- #
# A heatmap peak is considered "already found" if an LLM clip covers at least
# this fraction of it — no point emitting a near-duplicate.
_PEAK_COVERED_FRACTION = 0.5


def _text_between(transcript, start_s: float, end_s: float) -> str:
    """Transcript text overlapping a time window, joined in order."""
    return " ".join(
        seg.text
        for seg in transcript.segments
        if seg.end_s > start_s and seg.start_s < end_s and seg.text
    ).strip()


def _lift_to_scores(lift: float) -> ClipScores:
    """Derive 1-10 scores for a clip discovered purely from view data.

    A peak is direct evidence of hook potential and shareability — viewers
    rewatched it. It says nothing about whether the moment is self-contained,
    so completeness stays deliberately middling and lets the critic judge.
    """
    # lift 1.15 (the peak threshold) -> 6, lift 1.6+ -> 10.
    scaled = int(round(6 + (lift - 1.15) * 9))
    strong = max(1, min(10, scaled))
    return ClipScores(hook_potential=strong, completeness=5, shareability=strong)


def _fuse_heatmap(candidates: list[ClipCandidate], heatmap, transcript, max_clips: int):
    """Annotate candidates with retention and add clips for unfound peaks.

    Two distinct jobs:
      1. Every LLM clip gets its `retention_lift`, which feeds `ranked_score`
         so moments the audience actually rewatched rank higher.
      2. Peaks no LLM clip covers become candidates in their own right. This is
         what surfaces a great moment the transcript reads flat — and on a
         Whisper-transcribed video it's often the strongest signal available.
    """
    if heatmap is None or not heatmap.buckets:
        return candidates

    for c in candidates:
        c.retention_lift = heatmap.lift_for(c.start_s, c.end_s)

    peaks = heatmap.peaks(max_peaks=max_clips)
    added: list[ClipCandidate] = []
    next_id = len(candidates) + 1

    for peak in peaks:
        covered = any(
            (min(peak.end_s, c.end_s) - max(peak.start_s, c.start_s))
            >= _PEAK_COVERED_FRACTION * peak.duration_s
            for c in candidates
        )
        if covered:
            continue

        text = _text_between(transcript, peak.start_s, peak.end_s)
        if not text:
            # No words in this window (music, action, silence). We could still
            # cut it, but every downstream agent writes from transcript text,
            # so a textless clip would only invite hallucinated hooks.
            log.info(
                "clip_scout.peak_skipped_no_text",
                extra={"extra_fields": {"start_s": peak.start_s, "lift": peak.lift}},
            )
            continue

        added.append(
            ClipCandidate(
                clip_id=f"clip_{next_id}",
                start_s=peak.start_s,
                end_s=peak.end_s,
                transcript_text=text,
                reason_chosen=(
                    f"Most-replayed moment: viewers rewatched this "
                    f"{peak.lift:.1f}x more than the video average."
                ),
                scores=_lift_to_scores(peak.lift),
                origin="heatmap",
                retention_lift=peak.lift,
            )
        )
        next_id += 1

    if added:
        log.info(
            "clip_scout.heatmap_clips_added",
            extra={"extra_fields": {"added": len(added), "peaks_found": len(peaks)}},
        )
    return candidates + added


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #
def clip_scout_node(state: ContentState) -> dict:
    settings = get_settings()
    content_map = state.get("content_map")
    transcript = state.get("transcript")
    if content_map is None or transcript is None:
        return {
            "clip_candidates": [],
            "errors": ["clip_scout: no content map available"],
            "node_trace": ["clip_scout"],
        }

    round_ = state.get("critic_round", 0)
    max_clips = settings.max_clips
    usage: dict = {}

    if content_map.estimated_tokens <= CHUNK_THRESHOLD_TOKENS:
        # Small video: one call with the whole map (+ revision context if any).
        log.info("clip_scout.invoke", extra={"extra_fields": {"mode": "single", "round": round_}})
        picks, usage = _scout_call(
            content_map.render_for_prompt(), max_clips, _render_previous(state)
        )
    else:
        # Long video: map-reduce over chunks.
        chunks = _chunk_blocks(content_map.blocks)
        per_chunk = max(2, math.ceil(max_clips / len(chunks)) + 1)
        extra = _revision_note(state) if round_ > 0 else ""
        log.info(
            "clip_scout.invoke",
            extra={"extra_fields": {"mode": "chunked", "chunks": len(chunks),
                                    "per_chunk": per_chunk, "round": round_}},
        )
        picks = []
        for i, chunk in enumerate(chunks):
            cpicks, cusage = _scout_call(_render_blocks(chunk), per_chunk, extra)
            picks.extend(cpicks)
            usage = llm.merge_usage(usage, cusage)
            log.info("clip_scout.chunk",
                     extra={"extra_fields": {"chunk": i + 1, "of": len(chunks), "picks": len(cpicks)}})
        # Reduce: keep the best `max_clips` by self-reported score.
        picks.sort(key=_score, reverse=True)
        picks = picks[:max_clips]

    candidates: list[ClipCandidate] = []
    total_duration = transcript.total_duration_s
    for i, pick in enumerate(picks[:max_clips], start=1):
        start, end, notes = _validate_clip(pick, total_duration)
        if notes:
            log.info("clip_scout.clamped",
                     extra={"extra_fields": {"clip": f"clip_{i}", "adjustments": notes}})
        candidates.append(
            ClipCandidate(
                clip_id=f"clip_{i}",
                start_s=start,
                end_s=end,
                transcript_text=pick.transcript_text.strip(),
                reason_chosen=pick.reason_chosen.strip(),
                scores=ClipScores(
                    hook_potential=pick.hook_potential,
                    completeness=pick.completeness,
                    shareability=pick.shareability,
                ),
            )
        )

    candidates = _fuse_heatmap(
        candidates, state.get("heatmap"), transcript, max_clips
    )

    log.info("clip_scout.done",
             extra={"extra_fields": {
                 "candidates": len(candidates),
                 "from_heatmap": sum(1 for c in candidates if c.origin == "heatmap"),
                 "round": round_,
             }})
    return {
        "clip_candidates": candidates,
        "token_usage": {f"clip_scout.r{round_}": usage},
        "node_trace": ["clip_scout"],
    }
