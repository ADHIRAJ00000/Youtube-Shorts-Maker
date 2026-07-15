"""Title Smith agent.

Produces 5 title variants for the main video plus 3 titles + 2-3 thumbnail-text
options per approved clip. Main titles are capped at 60 chars — a corrective
retry fires on violation, with a hard trim as the backstop.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents import llm
from app.agents.state import (
    ClipTitleSet,
    ContentState,
    TitlePack,
    TitleSmithOutput,
)
from app.guardrails.guards import (
    cap_text,
    enforce_thumbnail_text,
    enforce_titles,
    title_violations,
)
from app.observability.logging_setup import get_logger

log = get_logger("app.agents.title_smith")

SYSTEM_PROMPT = """You are a YouTube title strategist. Given the full video \
overview and each approved clip: produce 5 title variants for the MAIN video \
(mix: how-to, listicle, curiosity, bold, keyword-forward for search) and 3 \
title variants per clip. Also produce 2-3 thumbnail text options per clip: 2-4 \
punchy words, ALL CAPS style, readable at small size. Titles must be truthful \
to content - no bait. Max 60 characters for main titles. Reference each clip by \
its exact clip_id. Output ONLY the structured schema."""


def title_smith_node(state: ContentState) -> dict:
    content_map = state.get("content_map")
    if content_map is None:
        return {"node_trace": ["title_smith"]}

    approved = state.get("approved_clips", []) or []
    overview = cap_text(content_map.render_for_prompt())
    if approved:
        clips_block = "\n".join(
            f'[{a.candidate.clip_id}] "{a.candidate.transcript_text[:200]}"'
            for a in approved
        )
    else:
        clips_block = "(no clips — short-form video; produce main video titles only)"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"VIDEO OVERVIEW:\n{overview}\n\nAPPROVED CLIPS:\n{clips_block}"),
    ]

    log.info("title_smith.invoke", extra={"extra_fields": {"clips": len(approved)}})
    out, usage = llm.structured_invoke(TitleSmithOutput, messages, temperature=0.7)

    # One corrective retry if a main title exceeds 60 chars. Only adopt the
    # retry if it actually returned titles — never let an empty/worse retry
    # clobber a good first response (the hard trim below is the real guarantee).
    if title_violations(out.main_video_titles):
        log.info("title_smith.retry", extra={"extra_fields": {"reason": "title > 60 chars"}})
        retry = messages + [
            HumanMessage(
                content="Some main titles exceeded 60 characters. Rewrite ALL main "
                "titles to be 60 characters or fewer."
            )
        ]
        out2, usage2 = llm.structured_invoke(TitleSmithOutput, retry, temperature=0.4)
        usage = llm.merge_usage(usage, usage2)
        if out2.main_video_titles:
            out = out2

    per_clip: dict[str, ClipTitleSet] = {}
    for cs in out.per_clip:
        per_clip[cs.clip_id] = ClipTitleSet(
            titles=enforce_titles(cs.titles)[:3],
            thumbnail_text=enforce_thumbnail_text(cs.thumbnail_text)[:3],
        )

    titles = TitlePack(
        main_video_titles=enforce_titles(out.main_video_titles)[:5],
        per_clip=per_clip,
    )
    log.info(
        "title_smith.done",
        extra={"extra_fields": {"main_titles": len(titles.main_video_titles), "clips": len(per_clip)}},
    )
    return {
        "titles": titles,
        "token_usage": {"title_smith": usage},
        "node_trace": ["title_smith"],
    }
