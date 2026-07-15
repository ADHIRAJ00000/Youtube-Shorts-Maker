"""SEO Packager agent.

Writes the video description (Markdown, with chapter timestamps derived from the
content map), up to 15 tags, up to 8 hashtags, and one Shorts + one Reels
caption. Tag/hashtag counts are enforced in code.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents import llm
from app.agents.state import ContentState, SEOOutput, SEOPack
from app.guardrails.guards import cap_text, enforce_hashtags, enforce_tags
from app.observability.logging_setup import get_logger
from app.tools.video_tools import seconds_to_hhmmss

log = get_logger("app.agents.seo_packager")

SYSTEM_PROMPT = """You are a YouTube SEO specialist. Produce: (1) a video \
description in Markdown - first 2 lines are the hook (visible before "show \
more"), then a 3-4 line summary, then chapter timestamps derived from the \
provided timeline (format: mm:ss Title), then a CTA line; (2) up to 15 tags \
mixing broad and specific keywords actually present in the content; (3) up to 8 \
hashtags; (4) one Shorts caption (YouTube style, <100 chars) and one Reels \
caption (Instagram style, may use line breaks + up to 3 emojis). Never stuff \
keywords unrelated to the actual content. Output ONLY the structured schema."""


# Cap chapters so a long video doesn't force an enormous SEO description that
# overruns the model's output-token budget.
_MAX_CHAPTERS = 20


def _timeline(content_map) -> str:
    """Compact timestamped outline the model can turn into chapters.

    For long videos the blocks are evenly sampled down to `_MAX_CHAPTERS` so the
    description stays a reasonable length.
    """
    blocks = list(content_map.blocks)
    if len(blocks) > _MAX_CHAPTERS:
        step = len(blocks) / _MAX_CHAPTERS
        blocks = [blocks[min(len(blocks) - 1, int(i * step))] for i in range(_MAX_CHAPTERS)]
    lines = []
    for b in blocks:
        preview = b.text[:70].strip()
        lines.append(f"{seconds_to_hhmmss(b.start_s)} {preview}")
    return "\n".join(lines)


def seo_packager_node(state: ContentState) -> dict:
    content_map = state.get("content_map")
    if content_map is None:
        return {"node_trace": ["seo_packager"]}

    titles = state.get("titles")
    main_title = ""
    if titles and titles.main_video_titles:
        main_title = titles.main_video_titles[0]

    overview = cap_text(content_map.render_for_prompt())
    human = (
        (f"WORKING TITLE: {main_title}\n\n" if main_title else "")
        + f"VIDEO OVERVIEW:\n{overview}\n\n"
        + f"TIMELINE (for chapters):\n{_timeline(content_map)}"
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human)]

    log.info("seo_packager.invoke")
    out, usage = llm.structured_invoke(SEOOutput, messages, temperature=0.5)

    seo = SEOPack(
        description_md=out.description_md.strip(),
        tags=enforce_tags(out.tags),
        hashtags=enforce_hashtags(out.hashtags),
        shorts_caption=out.shorts_caption.strip(),
        reels_caption=out.reels_caption.strip(),
    )
    log.info(
        "seo_packager.done",
        extra={"extra_fields": {"tags": len(seo.tags), "hashtags": len(seo.hashtags)}},
    )
    return {
        "seo": seo,
        "token_usage": {"seo_packager": usage},
        "node_trace": ["seo_packager"],
    }
