"""Structured JSON logging with a per-request/job id on every line.

A `ContextVar` carries the current job/request id through async call stacks,
so every log line emitted while handling a job is automatically tagged.
Use `bind_job_id(...)` at the start of a request/job.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

_job_id: ContextVar[str] = ContextVar("job_id", default="-")


def bind_job_id(job_id: str) -> None:
    """Attach a job/request id to all subsequent log lines in this context."""
    _job_id.set(job_id)


def current_job_id() -> str:
    return _job_id.get()


class _JsonFormatter(logging.Formatter):
    """Render each record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "job_id": _job_id.get(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any structured extras passed via `logger.info(..., extra={"extra_fields": {...}})`
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, idempotently."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    # Remove pre-existing handlers so reloads don't double-log.
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
