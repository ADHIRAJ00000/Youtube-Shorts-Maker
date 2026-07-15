"""Timestamp-aware chunking → ContentMap.

Merges fine-grained transcript segments into coarser ~30–60s logical blocks
(preferring to close a block on a sentence boundary). The resulting
`ContentMap` is what agents read — we never dump a raw multi-hour transcript
into a single prompt.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.ingestion.transcript import Transcript
from app.observability.logging_setup import get_logger

log = get_logger("app.ingestion.chunker")

# Rough token estimate: ~1.33 tokens per word for English.
_TOKENS_PER_WORD = 1.33
# Soft budget (in tokens) before switching to coarser blocks. The Clip Scout now
# chunks long transcripts itself (map-reduce), so we keep fine-grained ~45s
# blocks well past an hour of video for precise timestamps; only extremely long
# videos (>~2.5h) fall back to coarse blocks.
DEFAULT_TOKEN_BUDGET = 30_000

# Normal and coarse target block durations (seconds).
_TARGET_BLOCK_S = 45.0
_MAX_BLOCK_S = 60.0
_COARSE_TARGET_BLOCK_S = 120.0
_COARSE_MAX_BLOCK_S = 150.0

_SENTENCE_END = re.compile(r"[.!?]['\")\]]?\s*$")


class ContentBlock(BaseModel):
    """A merged, timestamped block of transcript text."""

    index: int
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    text: str
    word_count: int = Field(ge=0)

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)


class ContentMap(BaseModel):
    """The agent-facing view of a video: an ordered list of content blocks."""

    video_id: str
    total_duration_s: float = Field(ge=0)
    blocks: list[ContentBlock]
    total_words: int = Field(ge=0)
    estimated_tokens: int = Field(ge=0)
    is_coarse: bool = False

    def render_for_prompt(self) -> str:
        """Compact, timestamped rendering suitable for an LLM prompt."""
        lines = []
        for b in self.blocks:
            lines.append(f"[block {b.index} | {b.start_s:.1f}s-{b.end_s:.1f}s] {b.text}")
        return "\n".join(lines)


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END.search(text.strip()))


def build_content_map(
    transcript: Transcript,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> ContentMap:
    """Merge transcript segments into a `ContentMap`.

    If the transcript is large enough that the normal-granularity map would
    exceed `token_budget`, a coarser map (larger blocks) is produced instead.
    """
    total_words = transcript.word_count
    estimated_tokens = int(total_words * _TOKENS_PER_WORD)
    use_coarse = estimated_tokens > token_budget

    target = _COARSE_TARGET_BLOCK_S if use_coarse else _TARGET_BLOCK_S
    hard_max = _COARSE_MAX_BLOCK_S if use_coarse else _MAX_BLOCK_S

    blocks = _merge_segments(transcript, target, hard_max)

    content_map = ContentMap(
        video_id=transcript.video_id,
        total_duration_s=transcript.total_duration_s,
        blocks=blocks,
        total_words=total_words,
        estimated_tokens=estimated_tokens,
        is_coarse=use_coarse,
    )
    log.info(
        "content_map.built",
        extra={
            "extra_fields": {
                "video_id": transcript.video_id,
                "blocks": len(blocks),
                "total_words": total_words,
                "estimated_tokens": estimated_tokens,
                "is_coarse": use_coarse,
            }
        },
    )
    return content_map


def _merge_segments(
    transcript: Transcript,
    target_s: float,
    hard_max_s: float,
) -> list[ContentBlock]:
    """Greedily merge segments into blocks near `target_s`, closing on
    sentence boundaries where possible and never exceeding `hard_max_s`."""
    blocks: list[ContentBlock] = []
    if not transcript.segments:
        return blocks

    cur_start = transcript.segments[0].start_s
    cur_texts: list[str] = []
    cur_end = cur_start

    def flush(end_s: float) -> None:
        if not cur_texts:
            return
        text = " ".join(cur_texts).strip()
        if not text:
            return
        blocks.append(
            ContentBlock(
                index=len(blocks),
                start_s=round(cur_start, 3),
                end_s=round(end_s, 3),
                text=text,
                word_count=len(text.split()),
            )
        )

    for seg in transcript.segments:
        cur_texts.append(seg.text)
        cur_end = seg.end_s
        block_dur = cur_end - cur_start

        past_target = block_dur >= target_s
        over_max = block_dur >= hard_max_s
        # Close when we're past target AND at a sentence end, or forced by max.
        if over_max or (past_target and _ends_sentence(seg.text)):
            flush(cur_end)
            # Start a new block at the next segment.
            cur_texts = []
            cur_start = seg.end_s

    # Flush trailing content.
    if cur_texts:
        flush(cur_end)

    return blocks
