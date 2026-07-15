"""Hook Writer agent.

For each approved clip, writes exactly 3 hooks (curiosity, bold claim, direct
question), each <=12 words and grounded in the clip's actual content. A single
corrective retry fires if the model overshoots the word budget; a hard trim is
the backstop.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents import llm
from app.agents.state import ContentState, Hook, HooksOutput
from app.guardrails.guards import hook_violations, trim_hook
from app.observability.logging_setup import get_logger

log = get_logger("app.agents.hook_writer")

SYSTEM_PROMPT = """You write scroll-stopping opening lines for short-form \
videos. For each approved clip, write exactly 3 hooks in these styles: \
(1) curiosity (curiosity gap), (2) bold_claim (a bold claim), (3) question \
(a direct question). Each hook: max 12 words, no clickbait lies - every hook \
must be supported by the clip's actual content, no emojis. Reference each clip \
by its exact clip_id. Output ONLY the structured schema."""


def _all_hook_texts(out: HooksOutput) -> list[str]:
    return [h.text for hs in out.clip_hooks for h in hs.hooks]


def hook_writer_node(state: ContentState) -> dict:
    approved = state.get("approved_clips", []) or []
    if not approved:
        # Short-form path: no per-clip hooks.
        return {"hooks": {}, "node_trace": ["hook_writer"]}

    clips_block = "\n".join(
        f'[{a.candidate.clip_id}] "{a.candidate.transcript_text}"' for a in approved
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="APPROVED CLIPS:\n" + clips_block),
    ]

    log.info("hook_writer.invoke", extra={"extra_fields": {"clips": len(approved)}})
    out, usage = llm.structured_invoke(HooksOutput, messages, temperature=0.7)

    # One corrective retry if any hook exceeds the word budget.
    if hook_violations(_all_hook_texts(out)):
        log.info("hook_writer.retry", extra={"extra_fields": {"reason": "hook too long"}})
        retry = messages + [
            HumanMessage(
                content="Some hooks exceeded 12 words. Rewrite ALL hooks to be 12 words or fewer."
            )
        ]
        out2, usage2 = llm.structured_invoke(HooksOutput, retry, temperature=0.4)
        usage = llm.merge_usage(usage, usage2)
        if out2.clip_hooks:  # never adopt an empty/worse retry
            out = out2

    hooks: dict[str, list[Hook]] = {}
    for hs in out.clip_hooks:
        hooks[hs.clip_id] = [Hook(style=h.style, text=trim_hook(h.text)) for h in hs.hooks]

    log.info("hook_writer.done", extra={"extra_fields": {"clips_with_hooks": len(hooks)}})
    return {
        "hooks": hooks,
        "token_usage": {"hook_writer": usage},
        "node_trace": ["hook_writer"],
    }
