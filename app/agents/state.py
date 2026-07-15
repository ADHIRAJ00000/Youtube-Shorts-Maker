"""Shared LangGraph state schema + all structured domain models.

`ContentState` is the single object that flows through the agent graph. It is a
`TypedDict` (LangGraph's channel model) whose list/dict channels use reducers so
that parallel branches (hook writer + title smith) can append without clobbering
each other.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator

from app.ingestion.chunker import ContentMap
from app.ingestion.transcript import Transcript
from app.tools.video_tools import VideoMeta

ContentType = Literal["tutorial", "commentary", "podcast", "short_form", "unknown"]


def _coerce_str_list(v):
    """Coerce a list that may contain dicts into a list of plain strings.

    Smaller models sometimes return e.g. [{"title": "X"}] instead of ["X"];
    this recovers the string so structured parsing doesn't hard-fail.
    """
    if not isinstance(v, list):
        return v
    out: list[str] = []
    for x in v:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            val = (
                x.get("title") or x.get("text") or x.get("name") or x.get("value")
                or next((s for s in x.values() if isinstance(s, str)), None)
            )
            if val:
                out.append(str(val))
        elif x is not None:
            out.append(str(x))
    return out
Verdict = Literal["approve", "reject", "revise"]
HookStyle = Literal["curiosity", "bold_claim", "question"]


# --------------------------------------------------------------------------- #
# Clip pipeline models
# --------------------------------------------------------------------------- #
class ClipScores(BaseModel):
    """Per-clip quality scores (1–10 each)."""

    hook_potential: int = Field(ge=1, le=10)
    completeness: int = Field(ge=1, le=10)
    shareability: int = Field(ge=1, le=10)


class ClipCandidate(BaseModel):
    """A clip proposed by the Clip Scout."""

    clip_id: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    transcript_text: str
    reason_chosen: str
    scores: ClipScores

    @property
    def total_score(self) -> int:
        s = self.scores
        return s.hook_potential + s.completeness + s.shareability

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)


class CriticNote(BaseModel):
    """The Critic's verdict on one candidate."""

    clip_id: str
    verdict: Verdict
    reasons: list[str] = Field(default_factory=list)
    suggestions: Optional[str] = None


# --- LLM-boundary schemas (flattened for reliable structured output) -------- #
class ClipPick(BaseModel):
    """What the Clip Scout LLM returns per clip (scores flattened, no clip_id).

    The clip_id is assigned in code to guarantee uniqueness.
    """

    start_s: float = Field(ge=0, description="Clip start in seconds, from the content map")
    end_s: float = Field(ge=0, description="Clip end in seconds, from the content map")
    transcript_text: str = Field(description="The clip's transcript text")
    reason_chosen: str = Field(description="Why this segment makes a strong clip")
    hook_potential: int = Field(ge=1, le=10)
    completeness: int = Field(ge=1, le=10)
    shareability: int = Field(ge=1, le=10)


class ClipPickList(BaseModel):
    clips: list[ClipPick] = Field(default_factory=list)


class CriticReview(BaseModel):
    notes: list[CriticNote] = Field(default_factory=list)


class ApprovedClip(BaseModel):
    """A candidate that survived the critique loop, with its final rank."""

    candidate: ClipCandidate
    verdict: Verdict = "approve"
    rank: int = 0


# --------------------------------------------------------------------------- #
# Generation models
# --------------------------------------------------------------------------- #
class Hook(BaseModel):
    style: HookStyle
    text: str


class ClipTitleSet(BaseModel):
    titles: list[str] = Field(default_factory=list)
    thumbnail_text: list[str] = Field(default_factory=list)


class TitlePack(BaseModel):
    main_video_titles: list[str] = Field(default_factory=list)
    per_clip: dict[str, ClipTitleSet] = Field(default_factory=dict)


class SEOPack(BaseModel):
    description_md: str = ""
    tags: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list)
    shorts_caption: str = ""
    reels_caption: str = ""


# --- Generation LLM-boundary schemas ---------------------------------------- #
class HookSet(BaseModel):
    clip_id: str
    hooks: list[Hook] = Field(default_factory=list)


class HooksOutput(BaseModel):
    clip_hooks: list[HookSet] = Field(default_factory=list)


class ClipTitleSetLLM(BaseModel):
    clip_id: str
    titles: list[str] = Field(default_factory=list)
    thumbnail_text: list[str] = Field(default_factory=list)

    _coerce = field_validator("titles", "thumbnail_text", mode="before")(
        lambda v: _coerce_str_list(v)
    )


class TitleSmithOutput(BaseModel):
    main_video_titles: list[str] = Field(default_factory=list)
    per_clip: list[ClipTitleSetLLM] = Field(default_factory=list)

    _coerce = field_validator("main_video_titles", mode="before")(
        lambda v: _coerce_str_list(v)
    )


class SEOOutput(BaseModel):
    description_md: str = ""
    tags: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list)
    shorts_caption: str = ""
    reels_caption: str = ""

    _coerce = field_validator("tags", "hashtags", mode="before")(
        lambda v: _coerce_str_list(v)
    )


# --------------------------------------------------------------------------- #
# Reducers
# --------------------------------------------------------------------------- #
def _merge_dicts(a: dict, b: dict) -> dict:
    """Shallow-merge reducer for dict channels written by parallel branches."""
    out = dict(a)
    out.update(b)
    return out


# --------------------------------------------------------------------------- #
# The graph state
# --------------------------------------------------------------------------- #
class ContentState(TypedDict, total=False):
    # ---- input ----
    video_url: Optional[str]
    video_meta: Optional[VideoMeta]
    transcript: Optional[Transcript]
    content_map: Optional[ContentMap]
    content_type: ContentType

    # ---- clip pipeline ----
    clip_candidates: list[ClipCandidate]
    critic_feedback: list[CriticNote]
    critic_round: int
    approved_clips: list[ApprovedClip]

    # ---- generation ----
    hooks: dict[str, list[Hook]]  # clip_id -> hook variants
    titles: Optional[TitlePack]
    seo: Optional[SEOPack]

    # ---- output ----
    final_package: Optional[dict]

    # ---- control + observability ----
    errors: Annotated[list[str], operator.add]
    token_usage: Annotated[dict, _merge_dicts]
    trace_id: Optional[str]

    # ---- debugging: ordered record of executed nodes (reducer appends) ----
    node_trace: Annotated[list[str], operator.add]
