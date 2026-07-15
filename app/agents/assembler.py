"""Assembler agent.

Composes the final structured package (JSON) from everything the pipeline
produced, and renders a copy-paste-ready Markdown report to
`outputs/{video_id}/package.md`. This is the artifact the creator actually uses
for an upload, so the Markdown is organized for convenience.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agents.state import ContentState
from app.config import get_settings
from app.observability.logging_setup import get_logger
from app.tools.video_tools import seconds_to_hhmmss

log = get_logger("app.agents.assembler")


def _cost_summary(token_usage: dict[str, Any]) -> dict[str, Any]:
    total_tokens = sum(u.get("total_tokens", 0) for u in token_usage.values())
    total_cost = sum(u.get("cost_usd", 0.0) for u in token_usage.values())
    return {
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "llm_calls": len(token_usage),
        "per_agent": token_usage,
    }


def build_package(state: ContentState) -> dict[str, Any]:
    """Assemble the final package dict (pure — no file IO)."""
    transcript = state.get("transcript")
    content_map = state.get("content_map")
    approved = state.get("approved_clips", []) or []
    hooks = state.get("hooks", {}) or {}
    titles = state.get("titles")
    seo = state.get("seo")
    candidates = state.get("clip_candidates", []) or []
    feedback = {n.clip_id: n for n in (state.get("critic_feedback", []) or [])}

    video_id = transcript.video_id if transcript else "unknown"
    approved_ids = {a.candidate.clip_id for a in approved}

    clips_out = []
    for a in approved:
        c = a.candidate
        title_set = titles.per_clip.get(c.clip_id) if titles else None
        clips_out.append({
            "clip_id": c.clip_id,
            "rank": a.rank,
            "start_s": c.start_s,
            "end_s": c.end_s,
            "start": seconds_to_hhmmss(c.start_s),
            "end": seconds_to_hhmmss(c.end_s),
            "duration_s": c.duration_s,
            "total_score": c.total_score,
            "scores": c.scores.model_dump(),
            "reason_chosen": c.reason_chosen,
            "transcript_text": c.transcript_text,
            "verdict": a.verdict,
            "hooks": [h.model_dump() for h in hooks.get(c.clip_id, [])],
            "titles": title_set.titles if title_set else [],
            "thumbnail_text": title_set.thumbnail_text if title_set else [],
        })

    rejected_out = []
    for c in candidates:
        if c.clip_id in approved_ids:
            continue
        note = feedback.get(c.clip_id)
        rejected_out.append({
            "clip_id": c.clip_id,
            "start": seconds_to_hhmmss(c.start_s),
            "end": seconds_to_hhmmss(c.end_s),
            "total_score": c.total_score,
            "verdict": note.verdict if note else "unknown",
            "reasons": note.reasons if note else [],
            "transcript_text": c.transcript_text,
        })

    package = {
        "video": {
            "video_id": video_id,
            "video_url": state.get("video_url"),
            "language": transcript.language if transcript else None,
            "duration_s": transcript.total_duration_s if transcript else 0,
            "duration": seconds_to_hhmmss(transcript.total_duration_s) if transcript else "00:00",
            "content_type": state.get("content_type", "unknown"),
            "blocks": len(content_map.blocks) if content_map else 0,
        },
        "main_video_titles": titles.main_video_titles if titles else [],
        "approved_clips": clips_out,
        "rejected_clips": rejected_out,
        "seo": seo.model_dump() if seo else None,
        "critique": {
            "rounds": state.get("critic_round", 0),
            "proposed": len(candidates),
            "approved": len(approved),
            "rejected": len(rejected_out),
        },
        "cost": _cost_summary(state.get("token_usage", {}) or {}),
        "warnings": state.get("errors", []) or [],
    }
    return package


def render_markdown(pkg: dict[str, Any]) -> str:
    """Render the package as a clean, copy-paste-ready Markdown report."""
    v = pkg["video"]
    lines: list[str] = []
    lines.append(f"# Content Package — {v['video_id']}")
    if v.get("video_url"):
        lines.append(f"\n**Source:** {v['video_url']}  ")
    lines.append(
        f"**Duration:** {v['duration']} · **Type:** {v['content_type']} · "
        f"**Language:** {v.get('language')}"
    )
    crit = pkg["critique"]
    lines.append(
        f"**Clips:** {crit['approved']} approved / {crit['proposed']} proposed "
        f"({crit['rounds']} critique round(s))"
    )
    if pkg["warnings"]:
        lines.append(f"\n> ⚠️ {'; '.join(pkg['warnings'])}")

    # Main titles
    lines.append("\n## Main Video Titles\n")
    for t in pkg["main_video_titles"]:
        lines.append(f"- {t}")

    # Per-clip sections
    lines.append("\n## Clips\n")
    for c in pkg["approved_clips"]:
        lines.append(f"### #{c['rank']} · {c['start']}–{c['end']} ({c['duration_s']:.0f}s) — score {c['total_score']}/30\n")
        lines.append(f"> {c['transcript_text']}\n")
        lines.append(f"*Why:* {c['reason_chosen']}\n")
        if c["hooks"]:
            lines.append("**Hooks:**")
            for h in c["hooks"]:
                lines.append(f"- _{h['style']}_: {h['text']}")
            lines.append("")
        if c["titles"]:
            lines.append("**Clip titles:** " + " · ".join(c["titles"]))
        if c["thumbnail_text"]:
            lines.append("**Thumbnail text:** " + " · ".join(c["thumbnail_text"]))
        lines.append("")

    # Rejected
    if pkg["rejected_clips"]:
        lines.append("## Rejected / Dropped Clips\n")
        for r in pkg["rejected_clips"]:
            reasons = "; ".join(r["reasons"]) if r["reasons"] else "—"
            lines.append(f"- **{r['clip_id']}** ({r['start']}–{r['end']}) — {r['verdict']}: {reasons}")
        lines.append("")

    # SEO — copy-paste blocks
    seo = pkg["seo"]
    if seo:
        lines.append("## SEO — Copy & Paste\n")
        lines.append("### Description\n")
        lines.append("```")
        lines.append(seo["description_md"])
        lines.append("```\n")
        lines.append("### Tags\n")
        lines.append("```")
        lines.append(", ".join(seo["tags"]))
        lines.append("```\n")
        lines.append("### Hashtags\n")
        lines.append("```")
        lines.append(" ".join(seo["hashtags"]))
        lines.append("```\n")
        lines.append(f"**Shorts caption:** {seo['shorts_caption']}\n")
        lines.append("**Reels caption:**\n")
        lines.append("```")
        lines.append(seo["reels_caption"])
        lines.append("```\n")

    cost = pkg["cost"]
    lines.append("---")
    lines.append(
        f"_Generated by Content Repurposing Agent · {cost['total_tokens']} tokens · "
        f"${cost['total_cost_usd']:.4f} · {cost['llm_calls']} LLM calls_"
    )
    return "\n".join(lines)


def _write_outputs(pkg: dict[str, Any]) -> tuple[str, str]:
    settings = get_settings()
    video_id = pkg["video"]["video_id"]
    out_dir = Path(settings.output_dir) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "package.json"
    md_path = out_dir / "package.md"
    json_path.write_text(json.dumps(pkg, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(pkg), encoding="utf-8")
    return str(json_path), str(md_path)


def assembler_node(state: ContentState) -> dict:
    pkg = build_package(state)
    json_path, md_path = _write_outputs(pkg)
    pkg["outputs"] = {"json": json_path, "markdown": md_path}
    log.info(
        "assembler.done",
        extra={"extra_fields": {"video_id": pkg["video"]["video_id"], "md": md_path}},
    )
    return {"final_package": pkg, "node_trace": ["assembler"]}
