"""LangGraph construction: nodes, conditional edges, and the critique loop.

Topology
--------
    START → coordinator ─┬─(short-form)──────────────────► [hook_writer, title_smith]
                         └─(full)─► clip_scout → critic
    critic ─┬─(needs revision & round < MAX)─► clip_scout        ← the critique loop
            └─(approved / max rounds)──► select_clips ─► [hook_writer, title_smith]
    [hook_writer, title_smith] ─► seo_packager → assembler → END

`select_clips` (the approval gate) ranks approved clips and fans out to the
parallel generation pair, which the short-form route joins directly. Both paths
converge on `seo_packager` — a clean two-way join, no conditional-join deadlocks.
"""

from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph

from app.agents.assembler import assembler_node
from app.agents.clip_scout import clip_scout_node
from app.agents.coordinator import coordinator_node
from app.agents.critic import critic_node
from app.agents.hook_writer import hook_writer_node
from app.agents.select_clips import select_clips_node
from app.agents.seo_packager import seo_packager_node
from app.agents.state import ContentState
from app.agents.title_smith import title_smith_node
from app.observability.logging_setup import get_logger

log = get_logger("app.agents.graph")

DEFAULT_MAX_CRITIC_ROUNDS = 2

# The parallel generation branch both entry paths fan out to.
_GENERATION_FANOUT = ["hook_writer", "title_smith"]

NodeMap = dict[str, Callable[[ContentState], dict]]


def _resilient(name: str, fn: Callable[[ContentState], dict]) -> Callable[[ContentState], dict]:
    """Wrap a node so an exception degrades the run instead of crashing it.

    A failed agent records the error in state and yields no output of its own;
    downstream nodes handle the missing data (the pipeline always completes).
    """

    def wrapped(state: ContentState) -> dict:
        try:
            return fn(state)
        except Exception as exc:  # noqa: BLE001 — deliberate catch-all boundary
            log.error(
                "node.failed",
                extra={"extra_fields": {"node": name, "error": f"{type(exc).__name__}: {exc}"}},
            )
            return {
                "errors": [f"{name} failed: {type(exc).__name__}: {exc}"],
                "node_trace": [name],
            }

    return wrapped

DEFAULT_NODES: NodeMap = {
    "coordinator": coordinator_node,
    "clip_scout": clip_scout_node,
    "critic": critic_node,
    "select_clips": select_clips_node,
    "hook_writer": hook_writer_node,
    "title_smith": title_smith_node,
    "seo_packager": seo_packager_node,
    "assembler": assembler_node,
}


# --------------------------------------------------------------------------- #
# Routing functions
# --------------------------------------------------------------------------- #
def route_after_coordinator(state: ContentState) -> str | list[str]:
    """Short-form videos skip clip selection; everything else goes to the scout."""
    if state.get("content_type") == "short_form":
        log.info("route.coordinator", extra={"extra_fields": {"decision": "skip_clips"}})
        return _GENERATION_FANOUT
    log.info("route.coordinator", extra={"extra_fields": {"decision": "clip_scout"}})
    return "clip_scout"


def make_route_after_critic(max_rounds: int) -> Callable[[ContentState], str | list[str]]:
    """Build the post-critique router bound to a max-rounds budget."""

    def route_after_critic(state: ContentState) -> str:
        feedback = state.get("critic_feedback", []) or []
        needs_revision = any(n.verdict in ("reject", "revise") for n in feedback)
        rounds = state.get("critic_round", 0)

        if needs_revision and rounds < max_rounds:
            log.info(
                "route.critic",
                extra={"extra_fields": {"decision": "revise", "round": rounds}},
            )
            return "clip_scout"

        log.info(
            "route.critic",
            extra={"extra_fields": {"decision": "proceed", "round": rounds}},
        )
        return "select_clips"

    return route_after_critic


# --------------------------------------------------------------------------- #
# Graph builder
# --------------------------------------------------------------------------- #
def build_graph(
    max_critic_rounds: int = DEFAULT_MAX_CRITIC_ROUNDS,
    node_overrides: NodeMap | None = None,
):
    """Compile the agent graph.

    `node_overrides` lets tests swap in fake nodes (e.g. an always-reject critic)
    to exercise the critique loop deterministically.
    """
    nodes: NodeMap = {**DEFAULT_NODES, **(node_overrides or {})}

    g: StateGraph = StateGraph(ContentState)
    for name, fn in nodes.items():
        g.add_node(name, _resilient(name, fn))

    g.add_edge(START, "coordinator")
    g.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {"clip_scout": "clip_scout", "hook_writer": "hook_writer", "title_smith": "title_smith"},
    )
    g.add_edge("clip_scout", "critic")
    g.add_conditional_edges(
        "critic",
        make_route_after_critic(max_critic_rounds),
        {"clip_scout": "clip_scout", "select_clips": "select_clips"},
    )
    # Approval gate fans out to the parallel generation pair.
    g.add_edge("select_clips", "hook_writer")
    g.add_edge("select_clips", "title_smith")
    # Fan-in: both parallel branches converge on the SEO packager.
    g.add_edge("hook_writer", "seo_packager")
    g.add_edge("title_smith", "seo_packager")
    g.add_edge("seo_packager", "assembler")
    g.add_edge("assembler", END)

    return g.compile()
