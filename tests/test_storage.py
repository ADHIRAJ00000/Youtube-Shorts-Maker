"""Tests for the SQLite job store: lifecycle, dedup lookup, and stats."""

from __future__ import annotations

from pathlib import Path

from app.storage.jobs import JobStore


def _store(tmp_path: Path) -> JobStore:
    return JobStore(str(tmp_path / "jobs.sqlite"))


def test_job_lifecycle(tmp_path: Path) -> None:
    s = _store(tmp_path)
    jid = s.create_job(video_url="https://youtu.be/abc", video_id="abc")
    assert s.get_job(jid)["status"] == "queued"

    s.set_status(jid, "running")
    assert s.get_job(jid)["status"] == "running"

    s.finish_job(jid, clips_proposed=6, clips_approved=4, clips_rejected=2,
                 total_tokens=1000, cost_usd=0.01, package_path="/p/package.md",
                 result={"ok": True})
    job = s.get_job(jid)
    assert job["status"] == "done"
    assert job["clips_approved"] == 4
    assert job["result"] == {"ok": True}
    s.close()


def test_dedup_lookup(tmp_path: Path) -> None:
    s = _store(tmp_path)
    url = "https://youtu.be/xyz"
    assert s.find_completed_by_url(url) is None
    jid = s.create_job(video_url=url, video_id="xyz")
    # Not done yet -> not deduped.
    assert s.find_completed_by_url(url) is None
    s.finish_job(jid, clips_proposed=3, clips_approved=3, clips_rejected=0,
                 total_tokens=500, cost_usd=0.005, package_path="/p", result={})
    found = s.find_completed_by_url(url)
    assert found and found["id"] == jid
    s.close()


def test_stats_aggregate(tmp_path: Path) -> None:
    s = _store(tmp_path)
    for i in range(2):
        jid = s.create_job(video_url=f"u{i}", video_id=f"v{i}")
        s.finish_job(jid, clips_proposed=5, clips_approved=3, clips_rejected=2,
                     total_tokens=1000, cost_usd=0.01, package_path="/p", result={})
    failed = s.create_job(video_url="bad", video_id="bad")
    s.fail_job(failed, "boom")

    st = s.stats()
    assert st["videos_processed"] == 2
    assert st["jobs_failed"] == 1
    assert st["clips_proposed"] == 10
    assert st["clips_approved"] == 6
    assert st["approval_rate"] == 0.6
    assert st["avg_clips_per_video"] == 3.0
    assert st["avg_cost_per_video_usd"] == 0.01
    s.close()
