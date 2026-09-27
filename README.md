# ThinkTank

[![GitHub repo](https://img.shields.io/badge/GitHub-sandoracs%2Fthinktank-181717?logo=github)](https://github.com/sandoracs/thinktank)

Multi-party debate system where AI agents and humans talk around a virtual table
on a topic. Everyone hears everyone; agents have three-layer memory and a
fixed/evolving persona; a web hub configures and follows the session.

## Install

Pick one:

### 1. Source checkout (development) — requires [uv](https://docs.astral.sh/uv/)

```sh
git clone <this repo> thinktank && cd thinktank
uv sync            # installs Python 3.12 + all deps into .venv
# or simply:
./start.sh         # web hub on http://127.0.0.1:8080 (creates .env if missing)
```

### 2. Docker (no Python/uv needed on the host)

```sh
docker build -t thinktank .
docker run -d --name thinktank \
  -p 8080:8080 \
  -v thinktank-data:/app/data \
  -v "$PWD/.env":/app/.env:ro \
  thinktank
# → http://127.0.0.1:8080   (data volume keeps the DB; .env supplies your keys)
```

Non-root, healthcheck included; the DB lives in the mounted `/app/data`
volume, and the image itself contains no secrets (`.env`/`data/` are
`.dockerignore`d).

### 3. Installable command from git (`uv tool`, no PyPI account needed)

```sh
uv tool install --from git+<this repo url> thinktank
thinktank serve        # on PATH, isolated venv, no project checkout
```

## Verify

```sh
uv run pytest -q                 # 107 tests
uv run ruff check src tests      # lint
uv run pyright                  # strict type-check (0 errors)
```

## Run headless (research)

Create a session YAML (see `templates/sessions/llm_coauthorship.yaml` for a
full example), then:

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

- **Export** — analysis-ready event stream of the debate (API and CLI: see
  [API](#api)).
- **Approvals** — APPROVED drift mode turns a reflection proposal into a
  pending request; decide from the live table or the API (see [API](#api)).
- **Download** — the full session (config + complete event stream) as a
  single JSON file; the "Download" button in the live table, or the API
  (see [API](#api)).

## Remote agents (M7)

An external service can join the table over a fixed HTTP protocol. Point a
`ParticipantRef(remote=...)` seat at a base URL; the engine posts the context
and takes the reply as the message (protocol: see [API](#api)).

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

Create, edit, and delete agent templates from the library (`/agents`) —
every agent card is clickable and opens its editor. A small ✕ button in the
bottom-right corner deletes the agent; the app asks for confirmation first,
then redirects to the library. Endpoints: see [API](#api).

- **Persona age:** the persona core has a free-text `age` field ("Age" in
  the form); it renders into the system prompt (`Age: …`) and on the library card.
- **Color:** each agent gets one stable random color at creation (existing
  agents receive one at hub startup). The color tints the agent's transcript
  lines in the live table (the text color is chosen to stay readable on it)
  and appears as a dot in the sidebar and on the library cards.

## Session templates (save-as / prefill)

Save a builder form as a reusable session configuration, then prefill the
builder from any saved template; the builder lists saved templates and loads
one with the "Load" button. Endpoints: see [API](#api).

## Session edit and delete

A session that has not started yet can be edited after creation:
- **Live view:** the "Edit" button — enabled only while the status is
  `created` (dimmed while running/paused/ended).
- **Edit page:** the builder form prefilled with the stored configuration.
- **Semantics:** the stored config is replaced and the engine rebuilds from
  the latest config; the event history keeps every event, so an edited session
  still shows its full history under the newest configuration.
- After a session has run, **Reset** puts it back to `created`, so it can be
  edited and started again.
- **Delete:** the "Delete" button in the live table (asks for confirmation
  first; enabled only when the session is not running) — removes the session
  with its stored conversation (events, messages, persona history,
  approvals); agent memory is kept.

Endpoints: see [API](#api).

## Agent memory search (inspector)

Search an agent's episodic + long-term memory from the agent inspector
("memory search") over the hybrid `MemoryBackend`. Endpoint: see
[API](#api).

## Agent form: memory + initial state

The agent form (`/agents/new`) exposes the memory settings
(`working_window`, `summarize_every`, `retrieval_k`, `long_term`) and the
initial persona state (mood); both round-trip through the agent summary.

## API

Every HTTP endpoint the hub exposes (and the one protocol it calls out to),
grouped by resource.

### Sessions

- `POST /sessions` — create from the builder form (`save_as_template=1`,
  optionally `template_id`, to also save it as a session template).
- `GET /sessions/{id}/edit` → `POST /sessions/{id}/edit` — edit a session
  that has not started yet (prefilled builder form); replaces the stored
  config. `409` once the session has started, `404` for unknown sessions.
- `POST /api/sessions/{id}/reset` — wipe the conversation and restore the
  session to `created` so it can be edited/started again.
- `DELETE /api/sessions/{id}` — delete a non-running session and its stored
  conversation (agent memory is kept). `409` while running/paused, `404` for
  unknown sessions.
- `GET /sessions/{id}/download` — the full session (config + complete event
  stream) as a single JSON file.
- `GET /api/sessions/{id}/export?format=jsonl|csv` — analysis-ready event
  stream (CLI equivalent: `thinktank export <session-id> [--format jsonl|csv] [--out file]`).

### Approvals (APPROVED drift mode)

- `GET /api/sessions/{id}/approvals`
- `POST /api/sessions/{id}/approvals/{agent}` — body `{"decision": "approve"|"reject"}`.

### Agent memory

- `GET /api/sessions/{id}/agents/{agent}/memory?q=&k=&layer=` — search
  episodic/long-term memory (`layer`: `all` | `episodic` | `long_term`).

### Persona templates (agents)

- `GET /agents` — library. `GET /agents/new` → `POST /agents` — create.
  `GET /agents/{id}/edit` → `POST /agents/{id}` — edit (path and form id
  must match).
- `POST /api/agents`, `PUT /api/agents/{id}`, `DELETE /api/agents/{id}` —
  REST equivalents.

### Session templates

- `GET /api/session-templates`
- `GET/POST /api/session-templates[/{id}]`
- `DELETE /api/session-templates/{id}`

### Remote agent protocol (outbound)

Not served by the hub — the contract a remote participant's service must
implement; the engine calls it, not the other way around:

```
POST {url}/speak         -> {"content": "..."}
POST {url}/observe       -> {}
POST {url}/session_end   -> {}
```

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

The **"Default model"** setting is a select populated from these registered
providers (`<id>/<model>`); the current value is kept as a "(current)"
option even if it is not in the list, so saving never silently changes it.

### Per-agent output length

Each agent config can cap its output with `max_tokens` (YAML / API).
Omitted or `null` (the default) means **no cap** — the model writes to
natural length, so speeches are never truncated.

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
