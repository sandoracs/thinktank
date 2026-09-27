#!/usr/bin/env bash
# ThinkTank starter: ensures .env exists (Ollama defaults), checks the
# Ollama server, then launches the web hub or a headless session.
#
# Usage:
#   ./start.sh                          # web hub on http://127.0.0.1:8080
#   ./start.sh --port 9000              # custom port
#   ./start.sh --fake                   # offline FakeLLM, no API calls
#   ./start.sh run session.yaml         # headless run of a session template
#   ./start.sh run --fake session.yaml  # headless with FakeLLM
#
# Overrides (environment): OLLAMA_API_BASE, THINKTANK_DEFAULT_MODEL
set -euo pipefail
cd "$(dirname "$0")"

OLLAMA_API_BASE="${OLLAMA_API_BASE:-http://192.168.x.x:11434}"
MODEL="${THINKTANK_DEFAULT_MODEL:-ollama/qwen3.8}"
usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
HOST=127.0.0.1
PORT=8080
FAKE=0
MODE=serve
SESSION=""


while [ $# -gt 0 ]; do
  case "$1" in
    run)           MODE=run; shift ;;
    --port)        PORT="${2:?--port needs a value}"; shift 2 ;;
    --port=*)      PORT="${1#*=}"; shift ;;
    --fake)        FAKE=1; shift ;;
    -h|--help)     usage 0 ;;
    -*)            echo "unknown flag: $1" >&2; usage 1 ;;
    *)             if [ -n "$SESSION" ]; then echo "only one session YAML" >&2; exit 1
                   fi; SESSION="$1"; shift ;;
  esac
done
[ "$MODE" = run ] && [ -z "$SESSION" ] && { echo "run: missing session YAML (e.g. templates/sessions/llm_coauthorship.yaml)" >&2; exit 1; }

# --- prerequisites ----------------------------------------------------------
command -v uv >/dev/null || { echo "error: uv not found (https://docs.astral.sh/uv)" >&2; exit 1; }

if [ ! -f .env ]; then
  cat > .env <<EOF
# Created by start.sh — edit freely.
THINKTANK_DEFAULT_MODEL=$MODEL
OLLAMA_API_BASE=$OLLAMA_API_BASE
EOF
  echo "[start] wrote .env (model=$MODEL, ollama=$OLLAMA_API_BASE)"
fi

if [ "$FAKE" -eq 0 ]; then
  if ! tags=$(curl -sf -m 5 "$OLLAMA_API_BASE/api/tags"); then
    echo "error: Ollama not reachable at $OLLAMA_API_BASE" >&2
    echo "       fix OLLAMA_API_BASE, or retry with --fake (offline FakeLLM)" >&2
    exit 1
  fi
  name="${MODEL#ollama/}"
  if ! grep -q "\"name\":\"$name" <<<"$tags"; then
    echo "error: model '$name' not found on $OLLAMA_API_BASE" >&2
    echo "available:" >&2
    grep -o '"name":"[^"]*"' <<<"$tags" | sed 's/^"name":"//; s/"$//' | sed 's/^/  - /' >&2
    echo "       or override: THINKTANK_DEFAULT_MODEL=ollama/<name> ./start.sh" >&2
    exit 1
  fi
  echo "[start] ollama OK: $name @ $OLLAMA_API_BASE"
fi

# --- launch ------------------------------------------------------------------
FAKE_FLAG=""
if [ "$FAKE" = 1 ]; then FAKE_FLAG="--fake"; fi
if [ "$MODE" = serve ]; then
  echo "[start] hub: http://$HOST:$PORT"
  # Restart loop: the Settings page's "Save and restart" button sends
  # SIGTERM to the hub; this loop relaunches it with the freshly saved .env.
  trap 'exit 130' INT
  trap 'exit 143' TERM
  code=0
  while :; do
    uv run thinktank serve --host "$HOST" --port "$PORT" $FAKE_FLAG || code=$?
    echo "[start] hub stopped (exit $code) — restarting in 1 s; Ctrl-C: exit"
    sleep 1
  done
else
  [ -f "$SESSION" ] || { echo "error: session YAML not found: $SESSION" >&2; exit 1; }
  exec uv run thinktank run $FAKE_FLAG "$SESSION"
fi
