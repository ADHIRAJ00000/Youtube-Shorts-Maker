"""API tests: /process (file upload + URL dedup), /jobs/{id}, /stats, /health.

The LLM is mocked at the single seam so the background job runs the whole graph
offline. Ingestion uses a local transcript file (no network).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest
from fastapi.testclient import TestClient

from app.agents import llm as llm_mod
from app.agents.state import (
    ClipPick,
    ClipPickList,
    CriticNote,
    CriticReview,
    Hook,
    HookSet,
    HooksOutput,
    SEOOutput,
    TitleSmithOutput,
    ClipTitleSetLLM,
)

USAGE = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "cost_usd": 0.001}


def _dispatcher() -> Callable:
    def _fake(schema, messages, **kwargs):
        if schema is ClipPickList:
            return ClipPickList(clips=[
                ClipPick(start_s=0, end_s=30, transcript_text="A self contained clip about football.",
                         reason_chosen="strong", hook_potential=9, completeness=8, shareability=8),
                ClipPick(start_s=120, end_s=155, transcript_text="Another complete thought about AI.",
                         reason_chosen="good", hook_potential=7, completeness=7, shareability=7),
            ]), USAGE
        if schema is CriticReview:
            return CriticReview(notes=[
                CriticNote(clip_id="clip_1", verdict="approve", reasons=["clean"]),
                CriticNote(clip_id="clip_2", verdict="approve", reasons=["clean"]),
            ]), USAGE
        if schema is HooksOutput:
            return HooksOutput(clip_hooks=[
                HookSet(clip_id="clip_1", hooks=[
                    Hook(style="curiosity", text="You won't believe this"),
                    Hook(style="bold_claim", text="This changes football"),
                    Hook(style="question", text="Did you know this?"),
                ]),
            ]), USAGE
        if schema is TitleSmithOutput:
            return TitleSmithOutput(
                main_video_titles=["Football Meets AI", "The xG Revolution"],
                per_clip=[ClipTitleSetLLM(clip_id="clip_1", titles=["Clip One"], thumbnail_text=["BIG NEWS"])],
            ), USAGE
        if schema is SEOOutput:
            return SEOOutput(description_md="Hook\nLine\n\n00:00 Intro", tags=["football", "ai"],
                             hashtags=["#football"], shorts_caption="short", reels_caption="reels"), USAGE
        raise AssertionError(f"unexpected schema {schema}")
    return _fake


def _write_transcript(tmp_path: Path) -> Path:
    rows = [{"text": f"Sentence {i} about football and AI.", "start": i * 10, "duration": 10}
            for i in range(60)]  # 600s -> full pipeline
    p = tmp_path / "myvideo.json"
    p.write_text(json.dumps(rows))
    return p


def test_health() -> None:
    with TestClient(__import__("app.main", fromlist=["app"]).app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_process_file_upload_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_mod, "structured_invoke", _dispatcher())
    from app.main import app

    transcript = _write_transcript(tmp_path)
    with TestClient(app) as client:
        with transcript.open("rb") as fh:
            r = client.post("/process", files={"file": ("myvideo.json", fh, "application/json")})
        assert r.status_code == 200
        job_id = r.json()["job_id"]

        # Background task runs synchronously in the TestClient before returning.
        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "done", job.get("error")
        pkg = job["result"]
        assert len(pkg["approved_clips"]) == 2
        assert pkg["main_video_titles"]
        assert pkg["seo"]["tags"]

        # The Markdown report was written.
        assert Path(pkg["outputs"]["markdown"]).exists()

        stats = client.get("/stats").json()
        assert stats["videos_processed"] >= 1
        assert stats["approval_rate"] is not None


def test_process_url_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.main import app

    with TestClient(app) as client:
        # Seed a completed job for this URL directly in the store.
        from app.main import store
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        jid = store().create_job(video_url=url, video_id="dQw4w9WgXcQ")
        store().finish_job(jid, clips_proposed=3, clips_approved=2, clips_rejected=1,
                           total_tokens=100, cost_usd=0.001, package_path="/p", result={})

        r = client.post("/process", json={"video_url": url})
        assert r.status_code == 200
        body = r.json()
        assert body["deduplicated"] is True
        assert body["job_id"] == jid


def test_process_bad_video_url_422(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/process", json={"video_url": "https://example.com/nope"})
        assert r.status_code == 422


def test_process_unsupported_content_type_415() -> None:
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/process", content="raw", headers={"content-type": "text/plain"})
        assert r.status_code == 415


def test_get_missing_job_404() -> None:
    from app.main import app
    with TestClient(app) as client:
        assert client.get("/jobs/job_nope").status_code == 404


def test_ui_served_at_root() -> None:
    from app.main import app
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Content Repurposing Agent" in r.text
