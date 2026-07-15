"""Application configuration.

Loads settings from environment (and a local `.env` file) via Pydantic
Settings. Required secrets fail loudly at import/startup time so that a
misconfigured deployment never reaches the agent graph.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, sourced from env / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM ----
    llm_provider: Literal["groq", "openai", "anthropic"] = "groq"
    llm_api_key: str = Field(..., min_length=1, description="LLM provider API key")
    llm_model: str = "llama-3.3-70b-versatile"
    # Cheaper model used automatically when the primary is rate-limited (free
    # tier has per-model daily token caps). Set empty to disable fallback.
    llm_fallback_model: str = "llama-3.1-8b-instant"

    # ---- Langfuse (optional) ----
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ---- Pipeline tuning ----
    max_clips: int = Field(default=6, ge=1, le=12)
    max_critic_rounds: int = Field(default=2, ge=0, le=5)

    # ---- Storage ----
    db_path: str = "jobs.sqlite"
    output_dir: str = "outputs"

    @field_validator("llm_api_key")
    @classmethod
    def _no_placeholder_key(cls, v: str) -> str:
        if v.strip().lower().endswith("_here") or "your_" in v.lower():
            raise ValueError(
                "LLM_API_KEY looks like the placeholder from .env.example — "
                "set a real key."
            )
        return v

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Raises a clear, actionable error if required env vars are missing.
    """
    try:
        return Settings()  # type: ignore[call-arg]
    except Exception as exc:  # pydantic ValidationError or ValueError
        raise RuntimeError(
            "Configuration error — check your environment / .env file.\n"
            f"  Details: {exc}\n"
            "  Hint: copy .env.example to .env and fill in LLM_API_KEY."
        ) from exc
