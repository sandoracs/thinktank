# ThinkTank

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
| M1 | domain, events, `SessionEngine`, round-robin, `AIAgent`, LiteLLM/Fake, `thinktank run` | done |
| M2 | FastAPI hub, WebSocket live feed, human turns, pause/resume/stop, cost cap | done |
| M3 | memory: episodic/long-term, sqlite-vec + FTS5 + RRF, `EmbeddingProvider` | done |
| M4 | persona core/state, reflection, LOCKED/BOUNDED/SHADOW, inspector timeline | done |
| M5 | plugin registry (entry points), schema-driven forms, `HandRaisePriority`, `Bidding` | done |
| M6 | APPROVED mode + approval panel, JSONL/CSV export, JSON session download, consistency check, persona templates CRUD | done |
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
uv run thinktank run session.yaml --fake   # offline FakeLLM, no API calls
```

Without `--fake` a real LiteLLM model string is used (set `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, etc.). `--human-timeout` bounds how long a human turn waits.

## Run the web hub

```sh
uv run thinktank serve --port 8080     # real LLM (set your provider API keys)
uv run thinktank serve --fake          # offline: deterministic FakeLLM, no API calls
# open http://127.0.0.1:8080
```

Or, with the Ollama defaults baked in (creates `.env` if missing):

```sh
./start.sh                        # web hub
./start.sh run session.yaml       # headless session
./start.sh --fake                 # offline FakeLLM
```

Real LLM: put provider keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) in `.env`
(see `.env.example`) or the environment. The default model is
`claude-3-5-sonnet-latest` (`THINKTANK_DEFAULT_MODEL`); each agent can override
it with any LiteLLM model string (e.g. `ollama/qwen2.5` for a local Ollama).

The boardroom lists sessions; the builder creates one; the live table shows the
transcript with pause/resume/stop, human input, and (APPROVED mode) an approval
panel.

## M6: export, approvals, download

- **Export** — analysis-ready event stream:
  - API: `GET /api/sessions/{id}/export?format=jsonl|csv`
  - CLI: `thinktank export <session-id> [--format jsonl|csv] [--out file]`
- **Approvals** — APPROVED drift mode turns a reflection proposal into a pending
  request; decide from the live table or the API:
  - `GET  /api/sessions/{id}/approvals`
  - `POST /api/sessions/{id}/approvals/{agent}`  body `{"decision": "approve"|"reject"}`
- **Download** — the full session (config + complete event stream) as a single
  JSON file; the "Download" button in the live table, or directly:
  `GET /sessions/{id}/download`

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
("run `thinktank reembed`"). The command recomputes every stored vector
(rebuilding the vector table if the dimension changed) and records the new
model:

```sh
uv run thinktank reembed
```

## Consistency check (M6, opt-in)

Per-agent (`consistency_check: true`): a judge model scores the candidate
speech against the frozen persona core (1-5). Below `consistency_threshold`
the speech is regenerated once with the judge's feedback. Either way a
`ConsistencyViolation` event is recorded (shown in the agent inspector). Off
by default because it is an extra model call.

## Persona templates CRUD

Create, edit, and delete agent templates:
- **Library:** `GET /agents` — every agent card is clickable and opens its editor.
- **Create:** `GET /agents/new` → `POST /agents`
- **Edit:** `GET /agents/{id}/edit` → `POST /agents/{id}` (path and form id must match).
  A small ✕ button in the bottom-right corner deletes the agent; the app asks
  for confirmation first, then `DELETE /api/agents/{id}` and redirects to the library.
- **API:** `POST /api/agents`, `PUT /api/agents/{id}`, `DELETE /api/agents/{id}`
- **Persona age:** the persona core has a free-text `age` field ("Age" in
  the form); it renders into the system prompt (`Age: …`) and on the library card.
- **Color:** each agent gets one stable random color at creation (existing
  agents receive one at hub startup). The color tints the agent's transcript
  lines in the live table (the text color is chosen to stay readable on it)
  and appears as a dot in the sidebar and on the library cards.

## Session templates (save-as / prefill)

Save a builder form as a reusable session configuration, then prefill the
builder from any saved template (DESIGN §7 `session_templates`, §15 "save as template"):
- Builder: `POST /sessions` with `save_as_template=1` (optionally `template_id`);
  the builder lists saved templates and loads one with the "Load" button.
- API: `GET /api/session-templates`, `GET/POST /api/session-templates[/{id}]`,
  `DELETE /api/session-templates/{id}`.

## Session edit and delete

A session that has not started yet can be edited after creation (DESIGN §15 session editing):
- **Live view:** the "Edit" button — enabled only while the status is
  `created` (dimmed while running/paused/ended) → `GET /sessions/{id}/edit`.
- **Edit page:** the builder form prefilled with the stored configuration;
  "Save" → `POST /sessions/{id}/edit` (303 back to the live view).
- **Semantics:** the stored config is replaced and the engine rebuilds from
  the latest config; the event history keeps every event, so an edited session
  still shows its full history under the newest configuration. `409` once the session has started,
  `404` for unknown sessions.
- After a session has run, **Reset** (`POST /api/sessions/{id}/reset`) puts it
  back to `created`, so it can be edited and started again.
- **Delete:** the "Delete" button in the live table (asks for confirmation
  first; enabled only when the session is not running) →
  `DELETE /api/sessions/{id}` — removes the session with its stored
  conversation (events, messages, persona history, approvals); agent memory
  is kept. `409` while the session is running/paused (Stop it first),
  `404` for unknown sessions.

## Agent memory search (inspector)

Search an agent's episodic + long-term memory from the agent inspector
(DESIGN §15 "memory search"):
- `GET /api/sessions/{id}/agents/{agent}/memory?q=&k=&layer=` (layer:
  `all` | `episodic` | `long_term`) over the hybrid `MemoryBackend`.

## Agent form: memory + initial state

The agent form (`/agents/new`) exposes the memory settings
(`working_window`, `summarize_every`, `retrieval_k`, `long_term`) and the
initial persona state (mood); both round-trip through the agent summary.

## Configuration

All settings are env vars prefixed `THINKTANK_` (see `config.py`) or the
`.env` file. The **"Settings"** menu (next to the boardroom and persona
library) exposes every one of them:

- **Save** — validates and writes the values into `.env`. Settings marked
  "takes effect after a hub restart" (default model, database, host/port, …) take
  effect on the next hub start; the rest apply immediately (e.g.
  `llm_timeout_s`, `embedding_model` are re-read per call).
- **Save and restart** — same, then the hub restarts itself (the
  `start.sh` loop relaunches it) so every value is live right away.

### LLM providers (Settings block)

Besides env-var keys, the settings page has an **"LLM providers"** block:
register an endpoint (id, LiteLLM family, base URL, API key) and reference
its models anywhere as `<id>/<model>` (e.g. `myopenai/gpt-4o` in an agent
template or the default model). Keys are stored in
`<data_dir>/llm_providers.json` and masked on the page. Models without a
registered `<id>` fall through to the usual env-based resolution
(`OPENAI_API_KEY`, `OLLAMA_API_BASE`, …).


## Layout

```
src/thinktank/
  core/       engine, manager, bus, state (projection), context
  domain/     models, events (event sourcing)
  storage/    tables, repositories, db (pragmas, sqlite-vec, FTS5)
  llm/        client, fake, litellm_client
  memory/     embeddings, sqlite backend, distill
  persona/    policies, reflection, manager, prompts
  strategies/ base, round_robin, hand_raise, bidding
  plugins/    registry (entry-point loading)
  web/        FastAPI app, hub, pages, templates, static
  cli.py      thinktank run / serve / export / reembed
```

## Licensing

ThinkTank is **dual-licensed**. You may use it under **either** license, at your choice:

- **MIT License** — [`LICENSE`](./LICENSE). Free, permissive; the standard open track.
- **Commercial License** — [`COMMERCIAL-LICENSE.md`](./COMMERCIAL-LICENSE.md). For businesses
  that want a defined commercial relationship (term, support, SLA) alongside use.

The two licenses are independent: picking one does not grant or waive the other.
If your goal is to *restrict* commercial use or *require* source sharing, a permissive
MIT base does not do that — use a source-available or copyleft base (e.g. BSL/AGPL) plus
a commercial license instead.
