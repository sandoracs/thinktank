"""Application settings (pydantic-settings).

All values can be overridden via environment variables prefixed with
``THINKTANK_`` (e.g. ``THINKTANK_DATABASE_URL``) or a ``.env`` file in the
working directory. See ``.env.example``.
"""

import re
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the hub and CLI."""

    model_config = SettingsConfigDict(
        env_prefix="THINKTANK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage ---
    # Relative SQLite paths are resolved against ``data_dir``.
    database_url: str = Field(
        default="sqlite+aiosqlite:///./thinktank.db",
        description="SQLAlchemy async URL. Default keeps the DB in the cwd.",
    )
    data_dir: Path = Field(default=Path("data"), description="Default location for SQLite files.")

    # --- network / hub ---
    host: str = Field(default="127.0.0.1", description="Bind host; v1 stays loopback only.")
    port: int = Field(default=8080, ge=1, le=65535)

    # --- llm defaults ---
    default_model: str = Field(
        default="claude-3-5-sonnet-latest",
        description="Fallback LiteLLM model string when an agent template omits one.",
    )
    llm_timeout_s: float = Field(default=120.0, gt=0, description="Per-call timeout for LLM requests.")
    llm_max_concurrency: int = Field(default=8, ge=1, description="Global cap on concurrent LLM calls.")
    embedding_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        description="Multilingual embedding model for the ``litellm`` backend.",
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
    fake_llm: bool = Field(
        default=False,
        description="If true, use the offline FakeLLM instead of LiteLLM (``THINKTANK_FAKE_LLM=1``).",
    )
    # --- logging ---
    log_level: str = Field(default="INFO", description="structlog level name (DEBUG, INFO, ...).")
    log_json: bool = Field(default=True, description="Emit JSON logs instead of pretty console logs.")


def reload_settings() -> Settings:
    """Drop the cached settings and re-read env/``.env`` (call after writing the file)."""
    get_settings.cache_clear()
    return get_settings()


def write_env_settings(env_path: Path, values: dict[str, str]) -> None:
    """Merge field overrides into the ``THINKTANK_`` lines of ``env_path``.

    ``values`` maps a ``Settings`` field name to its raw env value. Existing
    lines (active or commented) are replaced in place so the file's comments
    survive; missing keys are appended at the end.
    """
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    remaining = dict(values)
    out: list[str] = []
    for line in lines:
        match = re.match(r"^#?\s*(THINKTANK_([A-Z0-9_]+))=", line)
        if match and match.group(2).lower() in remaining:
            out.append(f"{match.group(1)}={remaining.pop(match.group(2).lower())}")
            continue
        out.append(line)
    if remaining:
        if out and out[-1].strip():
            out.append("")
        for key, value in remaining.items():
            out.append(f"THINKTANK_{key.upper()}={value}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (cached)."""
    return Settings()
