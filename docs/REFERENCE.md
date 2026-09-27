# ThinkTank reference

Technical detail behind the deployment guide in [`../README.md`](../README.md):
what the features do, every HTTP endpoint, the advanced settings, and the
source layout.

- [Features](#features)
- [API](#api)
- [Advanced configuration](#advanced-configuration)
- [Source layout](#source-layout)

## Features

### Export, approvals, download

- **Export** — analysis-ready event stream of the debate (API and CLI: see
  [API](#api)).
- **Approvals** — APPROVED drift mode turns a reflection proposal into a
  pending request; decide from the live table or the API (see [API](#api)).
- **Download** — the full session (config + complete event stream) as a
  single JSON file; the "Download" button in the live table, or the API
  (see [API](#api)).

### Remote agents

An external service can join the table over a fixed HTTP protocol. Point a
`ParticipantRef(remote=...)` seat at a base URL; the engine posts the context
and takes the reply as the message (protocol: see [API](#api)).

A failing `speak` is recorded as an `Error` + `TurnSkipped` (the debate
continues); `observe`/`session_end` are best-effort. See
`tests/integration/test_remote_session.py` for the mock-server test.

Remote seats are supplied alongside the session config, keyed by participant
id: a `remotes` object on `POST /api/sessions`, or a top-level `remotes:`
section in a session YAML for `thinktank run`.

### Re-embedding

When the configured embedding model changes, the hub signals it at startup
("run `thinktank reembed`"). The command recomputes every stored vector
(rebuilding the vector table if the dimension changed) and records the new
model:

```sh
uv run thinktank reembed
```

### Consistency check (opt-in)

Per-agent (`consistency_check: true`): a judge model scores the candidate
speech against the frozen persona core (1-5). Below `consistency_threshold`
the speech is regenerated once with the judge's feedback. Either way a
`ConsistencyViolation` event is recorded (shown in the agent inspector). Off
by default because it is an extra model call.

### Persona templates

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
- **Memory + initial state:** the agent form (`/agents/new`) exposes the
  memory settings (`working_window`, `summarize_every`, `retrieval_k`,
  `long_term`) and the initial persona state (mood); both round-trip through
  the agent summary.

### Session templates (save-as / prefill)

Save a builder form as a reusable session configuration, then prefill the
builder from any saved template; the builder lists saved templates and loads
one with the "Load" button. Endpoints: see [API](#api).

### Session edit and delete

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

### Agent memory search (inspector)

Search an agent's episodic + long-term memory from the agent inspector
("memory search") over the hybrid `MemoryBackend`. Endpoint: see [API](#api).

## API

Every HTTP endpoint the hub exposes (and the one protocol it calls out to),
grouped by resource.

### Sessions

- `POST /sessions` — create from the builder form (`save_as_template=1`,
  optionally `template_id`, to also save it as a session template).
- `POST /api/sessions` — create from JSON (a `SessionConfig` body; an
  optional top-level `remotes` object supplies remote participants).
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

## Advanced configuration

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

## Source layout

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
