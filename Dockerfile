# syntax=docker/dockerfile:1
# ThinkTank — web hub + headless CLI in one image.
#
# Build:   docker build -t thinktank .
# Run:     docker run --rm -p 8080:8080 -v thinktank-data:/app/data thinktank
#          (mount your .env:  -v /path/to/.env:/app/.env:ro)
#
# Data (SQLite DB, provider registry) lives in /app/data — mount a volume
# so it survives container recreation. The DB is WAL-mode SQLite; a bind
# mount on a local disk works fine.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_LINK_MODE=copy

# ---------------------------------------------------------------------------
# Build stage: resolve + install the locked dependency set, then the project.
# ---------------------------------------------------------------------------
FROM base AS builder
WORKDIR /app

# 1) dependencies only (layer cache: code changes don't re-install deps)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) the project itself (src/ is the package; hatchling builds the wheel)
COPY src/ ./src/
COPY templates/ ./templates/
COPY README.md LICENSE COMMERCIAL-LICENSE.md pyproject.toml ./
RUN uv sync --frozen --no-dev --no-editable
# ---------------------------------------------------------------------------
# Runtime: lean, non-root, venv on PATH (no uv needed at run time).
# ---------------------------------------------------------------------------
FROM base AS runtime
RUN useradd -m -u 10001 thinktank
WORKDIR /app

# Absolute venv path is identical in both stages (UV_PROJECT_ENVIRONMENT),
# so the copy is valid.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Default to a public bind so `docker run -p` works; override per call.
# Real environment variables always beat a mounted .env file.
ENV THINKTANK_HOST=0.0.0.0 \
    THINKTANK_PORT=8080 \
    THINKTANK_DATA_DIR=/app/data \
    PYTHONUNBUFFERED=1

RUN mkdir -p /app/data && chown -R thinktank:thinktank /app/data
USER thinktank
VOLUME ["/app/data"]
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"THINKTANK_PORT\",\"8080\")}/',timeout=2)" || exit 1

# Default: web hub. For headless runs override the command, e.g.:
#   docker run --rm -v ... thinktank run templates/sessions/llm_coauthorship.yaml
CMD ["thinktank", "serve", "--host", "0.0.0.0", "--port", "8080"]
