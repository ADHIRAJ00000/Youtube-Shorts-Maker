"""Langfuse tracing setup.

Optional and fail-open: when Langfuse keys are absent (or the SDK errors), every
function here no-ops so the pipeline runs unchanged. When enabled, a LangChain
`CallbackHandler` is handed to `graph.invoke`, which auto-creates one trace per
job with a child span per agent invocation — so each critique-loop iteration
(clip_scout / critic running again) appears as its own span in the trace tree.
"""

from __future__ import annotations

from typing import Any, Optional

from app.config import get_settings
from app.observability.logging_setup import get_logger

log = get_logger("app.observability.tracing")


def get_callback_handler(session_id: Optional[str] = None) -> Optional[Any]:
    """Return a Langfuse LangChain callback handler, or None if disabled."""
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            session_id=session_id,
        )
    except Exception as exc:  # SDK/version/network issue — never fatal
        log.warning("langfuse.handler_failed", extra={"extra_fields": {"error": str(exc)}})
        return None


def trace_config(session_id: Optional[str] = None) -> dict[str, Any]:
    """Build a LangGraph invoke config with tracing wired in (empty if disabled)."""
    handler = get_callback_handler(session_id=session_id)
    if handler is None:
        return {}
    return {
        "callbacks": [handler],
        "run_name": "content-repurpose-job",
        "metadata": {"langfuse_session_id": session_id or "adhoc"},
    }


def flush(config: dict[str, Any]) -> None:
    """Flush any Langfuse handler in `config` (safe to call always)."""
    for cb in config.get("callbacks", []) or []:
        try:
            cb.flush()
        except Exception as exc:  # noqa: BLE001
            log.warning("langfuse.flush_failed", extra={"extra_fields": {"error": str(exc)}})
