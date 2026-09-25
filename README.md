# Roundtable

Multi-party debate system where AI agents and humans talk around a virtual table
on a topic. Everyone hears everyone; agents have three-layer memory and a
fixed/evolving persona; a web hub configures and follows the session. Single
machine, SQLite. Full specification in [`DESIGN.md`](DESIGN.md) (Hungarian).

## Status

Milestones **M0–M6** are implemented and tested (53 tests, ruff + pyright
strict clean). M7 (remote agents, token streaming, tool-using agents) is
optional and not started.

| Milestone | Scope | Status |
|---|---|---|
| M0 | repo, uv, ruff/pyright/pytest, Settings, DB, pragmas, **sqlite-vec load** | done |
| M1 | domain, events, `SessionEngine`, round-robin, `AIAgent`, LiteLLM/Fake, `roundtable run` | done |
| M2 | FastAPI hub, WebSocket live feed, human turns, pause/resume/stop, cost cap | done |
| M3 | memory: episodic/long-term, sqlite-vec + FTS5 + RRF, `EmbeddingProvider` | done |
| M4 | persona core/state, reflection, LOCKED/BOUNDED/SHADOW, inspector timeline | done |
| M5 | plugin registry (entry points), schema-driven forms, `HandRaisePriority`, `Bidding` | done |
| M6 | APPROVED mode + approval panel, JSONL/CSV export, replay view | done |

## Install

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync            # installs Python 3.12 + all deps into .venv
```

## Verify

```sh
uv run pytest -q                 # 53 tests
uv run ruff check src tests      # lint
uv run --with pyright pyright    # strict type-check (0 errors)
```

## Run headless (research)

Create a session YAML (see `DESIGN.md` §16 for the full example), then:

```sh
uv run roundtable run session.yaml --fake   # offline FakeLLM, no API calls
```

Without `--fake` a real LiteLLM model string is used (set `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, etc.). `--human-timeout` bounds how long a human turn waits.

## Run the web hub

```sh
uv run roundtable serve --port 8080
# open http://127.0.0.1:8080
```

The dashboard lists sessions; the builder creates one; the live table shows the
transcript with pause/resume/stop, human input, and (APPROVED mode) an approval
panel.

## M6: export, approvals, replay

- **Export** — analysis-ready event stream:
  - API: `GET /api/sessions/{id}/export?format=jsonl|csv`
  - CLI: `roundtable export <session-id> [--format jsonl|csv] [--out file]`
- **Approvals** — APPROVED drift mode turns a reflection proposal into a pending
  request; decide from the live table or the API:
  - `GET  /api/sessions/{id}/approvals`
  - `POST /api/sessions/{id}/approvals/{agent}`  body `{"decision": "approve"|"reject"}`
- **Replay** — step a finished session event-by-event: `GET /sessions/{id}/replay`

## Configuration

Env vars are prefixed `ROUNDTABLE_` (see `config.py`). Key ones:
`ROUNDTABLE_DATABASE_URL`, `ROUNDTABLE_HOST`, `ROUNDTABLE_PORT`,
`ROUNDTABLE_DEFAULT_MODEL`, `ROUNDTABLE_EMBEDDING_BACKEND` (`fake`|`litellm`).

## Layout

```
src/roundtable/
  core/       engine, manager, bus, state (projection), context
  domain/     models, events (event sourcing)
  storage/    tables, repositories, db (pragmas, sqlite-vec, FTS5)
  llm/        client, fake, litellm_client
  memory/     embeddings, sqlite backend, distill
  persona/    policies, reflection, manager, prompts
  strategies/ base, round_robin, hand_raise, bidding
  plugins/    registry (entry-point loading)
  web/        FastAPI app, hub, pages, templates, static
  cli.py      roundtable run / serve / export / reembed
```
