"""Shared test fixtures.

Config is deliberately fail-loud on a missing LLM_API_KEY, so tests inject a
dummy key and reset the cached Settings around each test.
"""

from __future__ import annotations

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_API_KEY", "gsk_test_dummy_key")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    # Neutralize any real Langfuse keys from a developer's .env so tracing is
    # disabled by default; individual tests opt in explicitly.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    # Keep all test artifacts out of the repo.
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "jobs.sqlite"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
