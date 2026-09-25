# Roundtable

Multi-party debate system where AI agents and humans talk around a virtual table
on a topic. Everyone hears everyone; agents have three-layer memory and a
fixed/evolving persona; a web hub configures and follows the session. Single
machine, SQLite. Full specification in [`DESIGN.md`](DESIGN.md) (Hungarian).

## Status

Milestones **M0–M6** plus the **RemoteAgent** (M7) and the **M3 `reembed`**
command are implemented and tested (76 tests, ruff + pyright strict clean).
Token streaming and tool-using agents (remaining M7 scope) are not started.

| Milestone | Scope | Status |
|---|---|---|
| M0 | repo, uv, ruff/pyright/pytest, Settings, DB, pragmas, **sqlite-vec load** | done |
| M1 | domain, events, `SessionEngine`, round-robin, `AIAgent`, LiteLLM/Fake, `roundtable run` | done |
| M2 | FastAPI hub, WebSocket live feed, human turns, pause/resume/stop, cost cap | done |
| M3 | memory: episodic/long-term, sqlite-vec + FTS5 + RRF, `EmbeddingProvider` | done |
| M4 | persona core/state, reflection, LOCKED/BOUNDED/SHADOW, inspector timeline | done |
| M5 | plugin registry (entry points), schema-driven forms, `HandRaisePriority`, `Bidding` | done |
| M6 | APPROVED mode + approval panel, JSONL/CSV export, replay view, consistency check, agent library CRUD | done |
| M7 | RemoteAgent (external HTTP participant); token streaming + tool agents remain | remote agent done |

## Install

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync            # installs Python 3.12 + all deps into .venv
```

## Verify

```sh
uv run pytest -q                 # 76 tests
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

## Remote agents (M7)

An external service can join the table over a fixed HTTP protocol. Point a
`ParticipantRef(remote=...)` seat at a base URL; the engine posts the context
and takes the reply as the message:

```
POST {url}/speak         -> {"content": "..."}
POST {url}/observe       -> {}
POST {url}/session_end   -> {}
```

A failing `speak` is recorded as an `Error` + `TurnSkipped` (the debate
continues); `observe`/`session_end` are best-effort. See
`tests/integration/test_remote_session.py` for the mock-server test.

## Re-embedding (M3)

When the configured embedding model changes, the hub signals it at startup
("run `roundtable reembed`"). The command recomputes every stored vector
(rebuilding the vector table if the dimension changed) and records the new
model:

```sh
uv run roundtable reembed
```

## Consistency check (M6, opt-in)

Per-agent (`consistency_check: true`): a judge model scores the candidate
speech against the frozen persona core (1-5). Below `consistency_threshold`
the speech is regenerated once with the judge's feedback. Either way a
`ConsistencyViolation` event is recorded (shown in the agent inspector). Off
by default because it is an extra model call.

## Agent library CRUD

Create, update, and delete agent templates:
- Form: `GET /agents/new` → `POST /agents`
- API: `POST /api/agents`, `PUT /api/agents/{id}`, `DELETE /api/agents/{id}`

## Session templates (save-as / prefill)

Save a builder form as a reusable session configuration, then prefill the
builder from any saved template (DESIGN §7 `session_templates`, §15
„mentés sablonként"):
- Builder: `POST /sessions` with `save_as_template=1` (optionally `template_id`);
  the builder lists saved templates and loads one with the „Betöltés" button.
- API: `GET /api/session-templates`, `GET/POST /api/session-templates[/{id}]`,
  `DELETE /api/session-templates/{id}`.

## Agent memory search (inspector)

Search an agent's episodic + long-term memory from the agent inspector
(DESIGN §15 „memória-kereső"):
- `GET /api/sessions/{id}/agents/{agent}/memory?q=&k=&layer=` (layer:
  `all` | `episodic` | `long_term`) over the hybrid `MemoryBackend`.

## Agent form: memory + initial state

The agent form (`/agents/new`) exposes the memory settings
(`working_window`, `summarize_every`, `retrieval_k`, `long_term`) and the
initial persona state (mood); both round-trip through the agent summary.

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
