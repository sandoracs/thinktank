"""Application settings (pydantic-settings).

All values can be overridden via environment variables prefixed with
``ROUNDTABLE_`` (e.g. ``ROUNDTABLE_DATABASE_URL``) or a ``.env`` file in the
working directory. See ``.env.example``.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the hub and CLI."""

    model_config = SettingsConfigDict(
        env_prefix="ROUNDTABLE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage ---
    # Relative SQLite paths are resolved against ``data_dir``.
    database_url: str = Field(
        default="sqlite+aiosqlite:///./roundtable.db",
        description="SQLAlchemy async URL. Default keeps the DB in the cwd.",
    )
    data_dir: Path = Field(default=Path("data"), description="Default location for SQLite files.")

    # --- network / hub ---
    host: str = Field(default="127.0.0.1", description="Bind host; v1 stays loopback only.")
    port: int = Field(default=8080, ge=1, le=65535)
    admin_token: str | None = Field(
        default=None,
        description="Optional shared token guarding config endpoints when exposed on a LAN.",
    )

    # --- llm defaults ---
    default_model: str = Field(
        default="claude-3-5-sonnet-latest",
        description="Fallback LiteLLM model string when an agent template omits one.",
    )
    llm_timeout_s: float = Field(default=120.0, gt=0, description="Per-call timeout for LLM requests.")
    llm_max_concurrency: int = Field(default=8, ge=1, description="Global cap on concurrent LLM calls.")
    embedding_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        description="Multilingual embedding model for the ``litellm`` backend (DESIGN.md §11).",
    )
    embedding_backend: str = Field(
        default="fake",
        description='"fake" (deterministic, offline) or "litellm" (Ollama/hosted API).',
    )
    embedding_dim: int = Field(
        default=32,
        ge=8,
        description="Embedding vector dimension; must match the provider's output size.",
    )
    # --- logging ---
    log_level: str = Field(default="INFO", description="structlog level name (DEBUG, INFO, ...).")
    log_json: bool = Field(default=True, description="Emit JSON logs instead of pretty console logs.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (cached)."""
    return Settings()
