# ThinkTank

[![GitHub repo](https://img.shields.io/badge/GitHub-sandoracs%2Fthinktank-181717?logo=github)](https://github.com/sandoracs/thinktank)

Multi-party debate system where AI agents and humans talk around a virtual table
on a topic. Everyone hears everyone; agents have three-layer memory and a
fixed/evolving persona; a web hub configures and follows the session.

Single machine, SQLite, no external services. Technical detail — features, the
full HTTP API, advanced settings — lives in
[`docs/REFERENCE.md`](docs/REFERENCE.md).

## Deploy

Pick one.

### Docker (no Python on the host)

```sh
docker build -t thinktank .
docker run -d --name thinktank \
  -p 8080:8080 \
  -v thinktank-data:/app/data \
  -v "$PWD/.env":/app/.env:ro \
  thinktank
# → http://127.0.0.1:8080
```

Non-root, healthcheck included. The DB lives in the mounted `/app/data` volume,
and the image contains no secrets (`.env`/`data/` are `.dockerignore`d) — keys
come from the mounted `.env`.

### Source checkout — requires [uv](https://docs.astral.sh/uv/)

```sh
git clone https://github.com/sandoracs/thinktank.git && cd thinktank
uv sync                        # Python 3.12 + all deps into .venv
uv run thinktank serve         # web hub on http://127.0.0.1:8080
```

### Installable command from git

```sh
uv tool install --from git+https://github.com/sandoracs/thinktank.git thinktank
thinktank serve        # on PATH, isolated venv, no project checkout
```

## Configure

Settings are env vars prefixed `THINKTANK_`, read from the environment or a
`.env` file next to the app (see [`.env.example`](.env.example)). The minimum
is one provider key:

```sh
ANTHROPIC_API_KEY=sk-...          # or OPENAI_API_KEY, GEMINI_API_KEY, …
THINKTANK_DEFAULT_MODEL=claude-3-5-sonnet-latest
THINKTANK_HOST=127.0.0.1          # v1 stays loopback only
THINKTANK_PORT=8080
THINKTANK_DATABASE_URL=sqlite+aiosqlite:///./thinktank.db   # relative paths resolve against THINKTANK_DATA_DIR
```

Any LiteLLM model string works, per agent or as the default (e.g.
`ollama/qwen2.5` against a local Ollama). The running hub's **Settings** page
edits every setting and writes it back to `.env`; extra provider endpoints can
be registered there too — see
[advanced configuration](docs/REFERENCE.md#advanced-configuration).

## Run

```sh
uv run thinktank serve --port 8080     # web hub (needs provider keys)
uv run thinktank serve --fake          # offline: deterministic FakeLLM, no API calls
```

The boardroom lists sessions; the builder creates one; the live table shows the
transcript with pause/resume/stop, human input, and (APPROVED mode) an approval
panel.

Headless, for research runs — create a session YAML (see
[`templates/sessions/llm_coauthorship.yaml`](templates/sessions/llm_coauthorship.yaml)),
then:

```sh
uv run thinktank run session.yaml            # real models
uv run thinktank run session.yaml --fake     # offline, no API calls
```

`--human-timeout` bounds how long a human turn waits; `-q` suppresses the live
transcript.

For a local-Ollama setup, `./start.sh` wraps both modes: it writes an Ollama
`.env` if missing, checks the server and model are reachable, then launches
(`./start.sh`, `./start.sh run session.yaml`, `./start.sh --fake`). Override
the endpoint with `OLLAMA_API_BASE` / `THINKTANK_DEFAULT_MODEL`.

## Operate

```sh
uv run thinktank export <session-id> --format jsonl   # analysis-ready event stream
uv run thinktank reembed                              # after changing the embedding model
```

The hub warns at startup when stored memory vectors were produced by a
different embedding model than the one now configured; `reembed` recomputes
them.

## Verify

```sh
uv run pytest -q                 # 107 tests
uv run ruff check src tests      # lint
uv run pyright                   # strict type-check (0 errors)
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
