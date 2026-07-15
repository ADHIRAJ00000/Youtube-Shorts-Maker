"""FastAPI application entry point.

Endpoints:
  GET  /health        — liveness + config check
  POST /process       — start a job from a YouTube URL (JSON) or transcript file
                        (multipart); returns a job_id and runs the graph in the
                        background. Deduplicates completed URLs.
  GET  /jobs/{id}     — job status + the final package when done
  GET  /stats         — aggregate production stats (approval rate, cost, …)
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from app.config import get_settings
from app.ingestion.transcript import TranscriptError, extract_video_id
from app.observability.logging_setup import bind_job_id, get_logger, setup_logging
from app.pipeline import build_initial_state, job_metrics, run_pipeline
from app.storage.jobs import JobStore

setup_logging()
log = get_logger("app.main")

_store: JobStore | None = None
UPLOAD_DIR = Path("uploads")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def store() -> JobStore:
    assert _store is not None, "JobStore not initialized"
    return _store


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store
    settings = get_settings()
    _store = JobStore(settings.db_path)
    UPLOAD_DIR.mkdir(exist_ok=True)
    log.info(
        "startup",
        extra={"extra_fields": {
            "provider": settings.llm_provider, "model": settings.llm_model,
            "fallback_model": settings.llm_fallback_model,
            "db": settings.db_path, "langfuse": settings.langfuse_enabled,
        }},
    )
    yield
    if _store:
        _store.close()
    log.info("shutdown")


app = FastAPI(
    title="Content Repurposing Agent",
    description="Multi-agent pipeline: YouTube video -> clips, hooks, titles, SEO.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    req_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:12]}"
    bind_job_id(req_id)
    response = await call_next(request)
    response.headers["x-request-id"] = req_id
    return response


# --------------------------------------------------------------------------- #
# Background job runner
# --------------------------------------------------------------------------- #
def run_job(job_id: str, source: str, video_url: str | None) -> None:
    """Ingest + run the graph + persist results. Runs in a worker thread."""
    bind_job_id(job_id)
    s = store()
    s.set_status(job_id, "running")
    try:
        initial = build_initial_state(source, video_url=video_url)
    except TranscriptError as exc:
        log.warning("job.ingest_failed", extra={"extra_fields": {"error": str(exc)}})
        s.fail_job(job_id, f"ingestion failed: {exc}")
        return
    except Exception as exc:  # unexpected
        log.error("job.ingest_error", extra={"extra_fields": {"error": str(exc)}})
        s.fail_job(job_id, f"ingestion error: {type(exc).__name__}: {exc}")
        return

    try:
        final = run_pipeline(initial, job_id=job_id)
        m = job_metrics(final)
        s.finish_job(
            job_id,
            clips_proposed=m["clips_proposed"],
            clips_approved=m["clips_approved"],
            clips_rejected=m["clips_rejected"],
            total_tokens=m["total_tokens"],
            cost_usd=m["cost_usd"],
            package_path=m["package_path"],
            result=m["package"],
        )
        # Per-job cost summary line (spec format).
        rounds = m["critic_rounds"]
        log.info(
            f"{job_id} done — {m['clips_proposed']} clips proposed, "
            f"{m['clips_approved']} approved ({rounds} critic "
            f"round{'s' if rounds != 1 else ''}), {m['total_tokens']:,} tokens, "
            f"${m['cost_usd']:.4f}"
        )
    except Exception as exc:
        log.error("job.failed", extra={"extra_fields": {"error": str(exc)}})
        s.fail_job(job_id, f"pipeline error: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    """Serve the single-page web UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "langfuse_enabled": settings.langfuse_enabled,
    }


@app.post("/process")
async def process(request: Request, background: BackgroundTasks) -> dict:
    """Start a repurposing job from a URL (JSON) or a transcript file (multipart)."""
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        video_url = (body or {}).get("video_url")
        if not video_url:
            raise HTTPException(status_code=422, detail="Missing 'video_url'.")
        try:
            video_id = extract_video_id(video_url)
        except TranscriptError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Dedup: return the existing completed job for this URL.
        existing = store().find_completed_by_url(video_url)
        if existing:
            return {"job_id": existing["id"], "status": "done", "deduplicated": True}

        job_id = store().create_job(video_url=video_url, video_id=video_id)
        background.add_task(run_job, job_id, video_url, video_url)
        return {"job_id": job_id, "status": "queued", "deduplicated": False}

    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "filename"):
            raise HTTPException(status_code=422, detail="Missing 'file' upload.")
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{upload.filename}"
        dest.write_bytes(await upload.read())
        video_id = Path(upload.filename).stem
        job_id = store().create_job(video_url=None, video_id=video_id)
        background.add_task(run_job, job_id, str(dest), None)
        return {"job_id": job_id, "status": "queued", "deduplicated": False}

    raise HTTPException(
        status_code=415,
        detail="Send application/json {video_url} or multipart/form-data with a file.",
    )


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = store().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/stats")
async def stats() -> dict:
    return store().stats()
