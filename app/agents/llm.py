"""LLM access + structured-output helper with token/cost accounting.

Every agent talks to the model through `structured_invoke`, which:
  * builds a provider chat model (Groq by default, on the free tier),
  * requests a Pydantic-typed structured output (no free-text parsing),
  * and returns the parsed object alongside a usage/cost dict.

Keeping this as the single seam means tests can monkeypatch one function to
run the whole graph offline.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Type, TypeVar

from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel

from app.config import get_settings
from app.observability.logging_setup import get_logger

log = get_logger("app.agents.llm")

T = TypeVar("T", bound=BaseModel)

# Approx Groq pricing (USD per 1M tokens) — used for a *nominal* cost metric.
# The free tier bills $0; these let /stats show realistic cost-per-video numbers.
_PRICING: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.1-70b-versatile": (0.59, 0.79),
    "mixtral-8x7b-32768": (0.24, 0.24),
}


def cost_for(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = _PRICING.get(model, (0.0, 0.0))
    return round(input_tokens / 1e6 * in_rate + output_tokens / 1e6 * out_rate, 6)


def get_chat_model(temperature: float = 0.4, model: str | None = None):
    """Construct the provider chat model from settings."""
    settings = get_settings()
    model = model or settings.llm_model

    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        # ChatGroq reads GROQ_API_KEY from env; set it explicitly to be safe.
        os.environ.setdefault("GROQ_API_KEY", settings.llm_api_key)
        # Cap output modestly: Groq counts max_tokens toward the per-minute
        # budget, and the 8B fallback's cap is only 6k TPM. 2000 is plenty for
        # our structured outputs (SEO chapters are capped separately).
        return ChatGroq(
            model=model, temperature=temperature, api_key=settings.llm_api_key,
            max_tokens=2000,
        )

    raise NotImplementedError(
        f"LLM provider {settings.llm_provider!r} is not wired yet (only 'groq')."
    )


def usage_from_message(msg: BaseMessage | None, model: str) -> dict[str, Any]:
    """Extract token usage + nominal cost from an AIMessage."""
    meta = getattr(msg, "usage_metadata", None) or {}
    input_tokens = int(meta.get("input_tokens", 0))
    output_tokens = int(meta.get("output_tokens", 0))
    total = int(meta.get("total_tokens", input_tokens + output_tokens))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total,
        "cost_usd": cost_for(model, input_tokens, output_tokens),
    }


def merge_usage(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Sum two usage dicts (used when a guardrail retry adds a second call)."""
    return {
        "input_tokens": a.get("input_tokens", 0) + b.get("input_tokens", 0),
        "output_tokens": a.get("output_tokens", 0) + b.get("output_tokens", 0),
        "total_tokens": a.get("total_tokens", 0) + b.get("total_tokens", 0),
        "cost_usd": round(a.get("cost_usd", 0.0) + b.get("cost_usd", 0.0), 6),
    }


def _schema_hint(schema: Type[BaseModel]) -> SystemMessage:
    """A system message pinning exact field names for json_mode.

    Groq requires the word "JSON" to appear when json_object response format is
    used; this also communicates the schema (function_calling would do that via
    the tool definition, but Groq's function-calling validator is flaky on
    longer outputs, so we use json_mode).
    """
    return SystemMessage(
        content=(
            "You must respond with a single JSON object that exactly matches "
            "this JSON schema (use these exact field names, no extras):\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
    )


def _rate_limit_kind(exc: Exception) -> Optional[str]:
    """Classify a rate limit as 'daily' (TPD), 'minute' (TPM), or None."""
    s = str(exc).lower()
    if "rate_limit" not in s and type(exc).__name__ != "RateLimitError":
        return None
    if "per day" in s or "tpd" in s:
        return "daily"
    if "per minute" in s or "tpm" in s:
        return "minute"
    return "minute"  # default: treat as the transient (per-minute) kind


# A model hitting its *daily* cap is skipped for a long cooldown (self-heals at
# the next-day reset). Per-minute spikes are NOT cooled down — they're handled
# in-line by a short sleep + retry on the same (more capable) model, because
# the fallback model's smaller per-minute cap often can't fit the request.
_MODEL_COOLDOWN: dict[str, float] = {}
_DAILY_COOLDOWN_S = 1800.0
_TPM_SLEEP_S = 12.0
_TPM_MAX_RETRIES = 3


def _model_available(model: str) -> bool:
    return time.time() >= _MODEL_COOLDOWN.get(model, 0.0)


def _mark_rate_limited(model: str, cooldown_s: float = _DAILY_COOLDOWN_S) -> None:
    _MODEL_COOLDOWN[model] = time.time() + cooldown_s


def structured_invoke(
    schema: Type[T],
    messages: list[BaseMessage],
    *,
    temperature: float = 0.4,
    model: str | None = None,
    _retries: int = 1,
) -> tuple[T, dict[str, Any]]:
    """Invoke the model for a Pydantic-typed result (Groq json_mode).

    Tries the primary model, then (on a rate-limit) the configured fallback
    model, retrying once on malformed output. Returns (parsed, usage_dict).
    """
    settings = get_settings()
    primary = model or settings.llm_model
    models = [primary]
    if settings.llm_fallback_model and settings.llm_fallback_model != primary:
        models.append(settings.llm_fallback_model)

    # Prefer models not in cooldown; if all are cooling down, still try the last.
    candidates = [m for m in models if _model_available(m)] or [models[-1]]

    full_messages = [_schema_hint(schema), *messages]
    last_err: Any = None
    # Accumulate usage across ALL attempts (failed parses/retries burn tokens
    # too) so our cost matches what the provider / Langfuse actually bills.
    acc_usage: dict[str, Any] = {}

    for m in candidates:
        llm = get_chat_model(temperature=temperature, model=m)
        structured = llm.with_structured_output(schema, method="json_mode", include_raw=True)
        parse_attempts = _retries + 1
        tpm_retries_left = _TPM_MAX_RETRIES
        attempt = 0
        advance_to_next_model = False
        while attempt < parse_attempts and not advance_to_next_model:
            try:
                result = structured.invoke(full_messages)
            except Exception as exc:  # provider/network error
                last_err = exc
                kind = _rate_limit_kind(exc)
                if kind == "daily":
                    _mark_rate_limited(m, _DAILY_COOLDOWN_S)
                    log.warning("structured_invoke.rate_limited_daily",
                                extra={"extra_fields": {"model": m, "schema": schema.__name__}})
                    advance_to_next_model = True  # this model is done for a while
                elif kind == "minute" and tpm_retries_left > 0:
                    tpm_retries_left -= 1
                    log.warning("structured_invoke.rate_limited_minute",
                                extra={"extra_fields": {"model": m, "sleep_s": _TPM_SLEEP_S}})
                    time.sleep(_TPM_SLEEP_S)  # let the per-minute window reset, retry same model
                elif kind == "minute":
                    advance_to_next_model = True  # exhausted TPM retries
                else:
                    attempt += 1  # non-rate-limit transient — consume a parse attempt
                continue

            acc_usage = merge_usage(acc_usage, usage_from_message(result.get("raw"), m))
            parsed = result.get("parsed")
            if parsed is not None:
                return parsed, acc_usage
            last_err = result.get("parsing_error")
            log.warning(
                "structured_invoke.parse_retry",
                extra={"extra_fields": {"schema": schema.__name__, "attempt": attempt, "model": m}},
            )
            attempt += 1

    raise ValueError(
        f"Structured output failed for {schema.__name__} on {models}: {last_err}"
    )
