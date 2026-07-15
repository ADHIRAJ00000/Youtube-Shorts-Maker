"""Critic agent — the heart of the agentic behavior.

Reviews each proposed clip against explicit quality criteria and issues a
verdict (approve / revise / reject). Reject and revise verdicts drive the
loop-back edge that sends the Clip Scout for another pass. A critic that never
rejects is decoration, so the prompt is deliberately strict.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents import llm
from app.agents.state import ContentState, CriticNote, CriticReview
from app.observability.logging_setup import get_logger

log = get_logger("app.agents.critic")

SYSTEM_PROMPT = """You are a ruthless content quality reviewer. You receive \
proposed clip candidates with their transcripts and scores. For EACH candidate, \
verify against these criteria: (1) Self-contained - no unresolved references; \
(2) Duration 20-60s; (3) Strong opening within the first 2 lines; (4) Complete \
ending - no mid-thought cutoffs; (5) Honest scoring - do the scores match the \
actual text? Verdicts: approve / revise (fixable, give exact suggestions) / \
reject (fundamentally weak, give reasons). Be strict: approving a weak clip \
wastes the creator's time. Do not rewrite clips yourself - critique only. \
Return one note per candidate, each referencing its exact clip_id. Output ONLY \
the structured schema."""


def _render_candidates(state: ContentState) -> str:
    candidates = state.get("clip_candidates", []) or []
    lines = []
    for c in candidates:
        lines.append(
            f"[{c.clip_id}] {c.start_s:.1f}s-{c.end_s:.1f}s "
            f"({c.duration_s:.0f}s) "
            f"scores(hook={c.scores.hook_potential}, "
            f"complete={c.scores.completeness}, share={c.scores.shareability}):\n"
            f'  "{c.transcript_text}"\n'
            f"  scout's reason: {c.reason_chosen}"
        )
    return "\n\n".join(lines)


def critic_node(state: ContentState) -> dict:
    round_ = state.get("critic_round", 0) + 1
    candidates = state.get("clip_candidates", []) or []

    if not candidates:
        log.info("critic.no_candidates")
        return {"critic_feedback": [], "critic_round": round_, "node_trace": ["critic"]}

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="CANDIDATES TO REVIEW:\n\n" + _render_candidates(state)),
    ]

    log.info("critic.invoke", extra={"extra_fields": {"critic_round": round_}})
    review, usage = llm.structured_invoke(CriticReview, messages, temperature=0.2)

    # Ensure every candidate has a note; default missing ones to approve so the
    # loop always terminates cleanly.
    noted = {n.clip_id: n for n in review.notes}
    notes: list[CriticNote] = []
    for c in candidates:
        notes.append(
            noted.get(c.clip_id)
            or CriticNote(clip_id=c.clip_id, verdict="approve", reasons=["no note returned"])
        )

    verdicts = {}
    for n in notes:
        verdicts[n.verdict] = verdicts.get(n.verdict, 0) + 1
    log.info(
        "critic.done",
        extra={"extra_fields": {"round": round_, "verdicts": verdicts}},
    )
    return {
        "critic_feedback": notes,
        "critic_round": round_,
        "token_usage": {f"critic.r{round_}": usage},
        "node_trace": ["critic"],
    }
