"""SQLite job store: history, status, results, and aggregate stats.

Deliberately small and synchronous (stdlib `sqlite3`). The background job
runner and the API endpoints share one connection guarded by a lock; queries
are tiny so brief blocking in the async endpoints is acceptable.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

JobStatus = Literal["queued", "running", "done", "failed"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id             TEXT PRIMARY KEY,
    video_url      TEXT,
    video_id       TEXT,
    status         TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    finished_at    TEXT,
    clips_proposed INTEGER DEFAULT 0,
    clips_approved INTEGER DEFAULT 0,
    clips_rejected INTEGER DEFAULT 0,
    total_tokens   INTEGER DEFAULT 0,
    cost_usd       REAL DEFAULT 0.0,
    package_path   TEXT,
    error          TEXT,
    result_json    TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ---- writes ---------------------------------------------------------- #
    def create_job(self, video_url: Optional[str], video_id: Optional[str]) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, video_url, video_id, status, created_at) "
                "VALUES (?, ?, ?, 'queued', ?)",
                (job_id, video_url, video_id, _now()),
            )
            self._conn.commit()
        return job_id

    def set_status(self, job_id: str, status: JobStatus) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = ? WHERE id = ?", (status, job_id)
            )
            self._conn.commit()

    def fail_job(self, job_id: str, error: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status='failed', error=?, finished_at=? WHERE id=?",
                (error, _now(), job_id),
            )
            self._conn.commit()

    def finish_job(
        self,
        job_id: str,
        *,
        clips_proposed: int,
        clips_approved: int,
        clips_rejected: int,
        total_tokens: int,
        cost_usd: float,
        package_path: str,
        result: dict[str, Any],
    ) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE jobs SET status='done', finished_at=?, clips_proposed=?,
                   clips_approved=?, clips_rejected=?, total_tokens=?, cost_usd=?,
                   package_path=?, result_json=? WHERE id=?""",
                (
                    _now(), clips_proposed, clips_approved, clips_rejected,
                    total_tokens, round(cost_usd, 6), package_path,
                    json.dumps(result), job_id,
                ),
            )
            self._conn.commit()

    # ---- reads ----------------------------------------------------------- #
    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def find_completed_by_url(self, video_url: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE video_url = ? AND status='done' "
                "ORDER BY finished_at DESC LIMIT 1",
                (video_url,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                """SELECT
                     SUM(CASE WHEN status='done'   THEN 1 ELSE 0 END) AS done,
                     SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed,
                     COUNT(*)                                         AS total,
                     COALESCE(SUM(clips_proposed),0)                  AS proposed,
                     COALESCE(SUM(clips_approved),0)                  AS approved,
                     COALESCE(SUM(clips_rejected),0)                  AS rejected,
                     COALESCE(SUM(cost_usd),0.0)                      AS total_cost,
                     COALESCE(SUM(total_tokens),0)                    AS total_tokens
                   FROM jobs"""
            ).fetchone()

        done = row["done"] or 0
        proposed = row["proposed"] or 0
        approved = row["approved"] or 0
        return {
            "videos_processed": done,
            "jobs_failed": row["failed"] or 0,
            "jobs_total": row["total"] or 0,
            "clips_proposed": proposed,
            "clips_approved": approved,
            "clips_rejected": row["rejected"] or 0,
            "approval_rate": round(approved / proposed, 4) if proposed else None,
            "rejection_rate": round((proposed - approved) / proposed, 4) if proposed else None,
            "avg_clips_per_video": round(approved / done, 2) if done else None,
            "avg_cost_per_video_usd": round((row["total_cost"] or 0.0) / done, 6) if done else None,
            "total_cost_usd": round(row["total_cost"] or 0.0, 6),
            "total_tokens": row["total_tokens"] or 0,
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        if d.get("result_json"):
            try:
                d["result"] = json.loads(d.pop("result_json"))
            except json.JSONDecodeError:
                d["result"] = None
        else:
            d.pop("result_json", None)
            d["result"] = None
        return d

    def close(self) -> None:
        with self._lock:
            self._conn.close()
