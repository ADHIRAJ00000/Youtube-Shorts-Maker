# syntax=docker/dockerfile:1
# ---------- Stage 1: builder ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build into an isolated virtualenv we can copy to the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Non-root user.
RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app ./app
COPY evals ./evals
COPY pytest.ini ./

# Writable dirs for outputs / uploads / sqlite, owned by the non-root user.
RUN mkdir -p outputs uploads && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

# Bind to $PORT if the host provides one (Render, Cloud Run, Fly…), else 8000.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD sh -c 'python -c "import os,urllib.request as u,sys; sys.exit(0 if u.urlopen(\"http://127.0.0.1:\"+os.environ.get(\"PORT\",\"8000\")+\"/health\").status==200 else 1)"'

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
