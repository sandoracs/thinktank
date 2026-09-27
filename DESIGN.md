# ThinkTank – Design Document (v1)

A multi-participant discussion system in which AI agents and humans talk about a topic around a virtual table. Everyone hears everyone else, the agents have memory, and their persona can be fixed or allowed to evolve in a controlled way. A web hub configures and monitors the system.

---

## 1. Goals and Non-Goals

**Goals (v1)**

- AI agents and human participants take part on equal terms, through the same interface.
- The moderator can be a human or an AI.
- The speaking order is swappable as a plugin; default: round robin.
- Three-tier memory per agent (working, episodic, long-term).
- A persona consists of a fixed core and a mutable state; how it may change is configurable per agent.
- Web hub: configuration, live monitoring, human contributions, agent inspector.
- Everything that happens is stored as an event: replayable and exportable for research analysis.
- Runs on a single machine (PC or Mac) with SQLite.

**Non-Goals (v1)**

- Postgres support, multi-user access management, internet-facing deployment.
- Token-level streaming to the UI (v2).
- Audio- or video-based participation.

---

## 2. Technology Stack

| Area | Choice | Notes |
|---|---|---|
| Language | Python 3.12+ | |
| Package and environment management | uv | let uv install Python as well (see 14, risks) |
| Web | FastAPI + uvicorn | REST + WebSocket |
| Schemas | Pydantic v2, pydantic-settings | config, events, API |
| Database | SQLite, SQLAlchemy 2.0 async + aiosqlite | WAL mode |
| Migrations | Alembic, `render_as_batch=True` | needed for SQLite column modifications |
| Vector search | sqlite-vec | `vec0` virtual table |
| Full-text search | SQLite FTS5 | hybrid search with the vector one |
| LLM | LiteLLM (`acompletion`, `aembedding`) | Claude, GPT, Gemini, Ollama behind one interface |
| Embedding | sentence-transformers or Ollama | **multilingual** model for Hungarian |
| UI | Jinja2 + HTMX (+ ws extension) | no build step |
| Charts | Chart.js, as static files | drift timeline |
| Retry | tenacity | |
| Logging | structlog | JSON logs |
| Quality | ruff, pyright strict, pytest + pytest-asyncio, pre-commit | |

---

## 3. Architecture

```
┌──────────────────── Web UI (Jinja2 + HTMX) ────────────────────┐
│   REST: configuration, control        WebSocket: live events   │
└───────────────────────────────┬────────────────────────────────┘
                           FastAPI Hub
     ┌──────────────┬───────────┴──────┬──────────────────┐
 SessionManager  PluginRegistry     EventBus         Repositories
 (running sessions) (entry points)   (in-process)     (SQLite)
     │
 SessionEngine ──► TurnStrategy (plugin)
     │
     ├──► ContextBuilder ──► MemoryBackend (plugin) ──► sqlite-vec + FTS5
     │
     ├──► Participants: AIAgent | HumanParticipant | RemoteAgent (plugin)
     │                     │
     │                  LLMClient (LiteLLM)
     │
     └──► PersonaManager ──► DriftPolicy (plugin), Reflection, ConsistencyCheck
```

**Principles**

1. **Event sourcing.** The `events` table is the source of truth. The `messages`, `persona_versions`, and similar tables are projections derived from it. The live UI, reconnection, replay, and export all build on the event stream.
2. **One interface for every participant.** The engine does not know whether it is a human or an AI that is speaking.
3. **Everything swappable is a plugin**, including the built-ins, all with the same registration mechanism.
4. **The LLM call is an injected dependency** (`LLMClient` protocol), so the core is fully testable with a `FakeLLM`, without any API calls.

---

## 4. Directory Structure

```
thinktank/
├── pyproject.toml
├── alembic.ini
├── .env.example
├── migrations/
├── templates/                   # saved session- and agent-templates (YAML)
│   ├── agents/
│   └── sessions/
├── src/thinktank/
│   ├── config.py                # Settings (pydantic-settings)
│   ├── cli.py                   # thinktank run / serve / export / reembed
│   ├── domain/
│   │   ├── models.py            # AgentConfig, PersonaCore, PersonaState, SessionConfig, Message
│   │   └── events.py            # event types
│   ├── core/
│   │   ├── engine.py            # SessionEngine
│   │   ├── manager.py           # SessionManager (multiple sessions, lifecycle)
│   │   ├── state.py             # SessionState + projection from events
│   │   ├── context.py           # ContextBuilder, token budget
│   │   └── bus.py               # EventBus (pub/sub, toward WS)
│   ├── participants/
│   │   ├── base.py              # Participant protocol
│   │   ├── ai_agent.py
│   │   ├── human.py
│   │   └── remote.py            # external agent connecting over HTTP (end of v1 / v2)
│   ├── strategies/
│   │   ├── base.py              # TurnStrategy ABC
│   │   ├── round_robin.py
│   │   └── hand_raise.py        # wrapper: hand-raising priority on top of any strategy
│   ├── memory/
│   │   ├── base.py              # MemoryBackend ABC
│   │   ├── sqlite_memory.py     # sqlite-vec + FTS5, RRF fusion
│   │   ├── embeddings.py        # EmbeddingProvider
│   │   └── summarizer.py        # episodic summaries
│   ├── persona/
│   │   ├── manager.py
│   │   ├── policies.py          # LOCKED, BOUNDED, APPROVED, FREE
│   │   ├── reflection.py
│   │   └── consistency.py
│   ├── llm/
│   │   ├── client.py            # LiteLLM wrapper: retry, timeout, cost, semaphore
│   │   ├── fake.py              # FakeLLM for tests
│   │   └── prompts/             # Jinja2 prompt templates
│   ├── plugins/
│   │   └── registry.py
│   ├── storage/
│   │   ├── db.py                # engine, pragmas, loading sqlite-vec
│   │   ├── tables.py            # ORM models
│   │   └── repositories.py
│   ├── api/
│   │   ├── app.py
│   │   ├── routes_library.py    # agent and session templates
│   │   ├── routes_sessions.py
│   │   ├── routes_plugins.py
│   │   ├── ws.py
│   │   └── schemas.py
│   └── web/
│       ├── templates/
│       └── static/
└── tests/
    ├── unit/
    ├── integration/
    └── live/                    # real API calls, skipped by default
```

---

## 5. Domain Model

### 5.1 Persona

```python
class PersonaCore(BaseModel, frozen=True):
    """Never changes during a session."""
    name: str
    role: str                          # e.g. "skeptical methodologist"
    expertise: list[str]
    values: list[str]
    communication_style: str           # e.g. "concise, asks back, gives examples"
    temperament: str
    background: str = ""
    boundaries: list[str] = []         # things it never does/says

class Stance(BaseModel):
    position: str                      # textual stance
    confidence: float = Field(ge=0, le=1)

class PersonaState(BaseModel):
    """May change per the drift policy."""
    stances: dict[str, Stance] = {}    # key: id of one of the session's debate questions
    attitudes: dict[str, float] = {}   # participant id -> trust (-1..1)
    mood: str = "neutral"
```

### 5.2 Drift Policy

```python
class DriftMode(StrEnum):
    LOCKED = "locked"      # state must not change
    BOUNDED = "bounded"    # may change, with a step-size limit
    APPROVED = "approved"  # proposal -> approved by a human/moderator
    FREE = "free"          # free; the core is still protected

class DriftConfig(BaseModel):
    mode: DriftMode = DriftMode.LOCKED
    max_confidence_delta: float = 0.2     # BOUNDED: per round
    max_attitude_delta: float = 0.3
    max_stance_changes_per_round: int = 1
    shadow_reflection: bool = False       # reflection runs even under LOCKED, just not applied
```

`shadow_reflection` is for research purposes: it measures what "pressure" a fixed persona is under without yielding to it.

### 5.3 Agent and Session Configuration

```python
class MemoryConfig(BaseModel):
    working_window: int = 12           # last N messages verbatim
    summarize_every: int = 8           # new-message count per episodic summary
    retrieval_k: int = 5
    long_term: bool = True             # memory across sessions

class AgentConfig(BaseModel):
    id: str                            # template id = long-term identity
    type: str = "llm"                  # name of the participant plugin
    model: str                         # LiteLLM model string
    temperature: float = 0.8
    max_tokens: int = 600
    persona: PersonaCore
    initial_state: PersonaState = PersonaState()
    drift: DriftConfig = DriftConfig()
    memory: MemoryConfig = MemoryConfig()
    consistency_check: bool = False
    carry_over_state: bool = False     # carry persona state into the next session

class DebateQuestion(BaseModel):
    id: str                            # e.g. "q_authorship"
    text: str                          # e.g. "Can an LLM be a co-author?"

class StrategyRef(BaseModel):
    name: str = "round_robin"
    params: dict[str, Any] = {}

class StopConditions(BaseModel):
    max_rounds: int | None = 6
    max_messages: int | None = None
    max_cost_usd: float | None = 2.0
    max_duration_s: int | None = None

class SessionConfig(BaseModel):
    title: str
    topic: str                         # the moderator's opening framing / research area
    questions: list[DebateQuestion]    # stances are measured along these
    participants: list[ParticipantRef] # agent template id or human (name)
    moderator: str | None              # participant id, or None
    strategy: StrategyRef = StrategyRef()
    stop: StopConditions = StopConditions()
    reflection_every_rounds: int = 1
    language: str = "en"
```

**Why do we need `questions`?** When stances are tied to predefined questions, drift is measurable and comparable: for every question, every agent, every round there is a `position` and a `confidence`. Free-text "opinion change" does not yield this reliably.

### 5.4 Message

```python
class Message(BaseModel):
    id: UUID
    session_id: UUID
    seq: int
    speaker_id: str
    kind: Literal["speech", "moderator", "system"]
    content: str
    reply_to: UUID | None = None
    meta: dict[str, Any] = {}          # model, tokens, cost, latency
```

---

## 6. Events

Every event: `session_id`, `seq` (strictly increasing within the session, allocated by the application), `type`, `payload` (JSON), `created_at` (UTC).

| Event | Payload essentials |
|---|---|
| `SessionCreated` | full `SessionConfig` snapshot |
| `SessionStarted` / `Paused` / `Resumed` | |
| `SessionEnded` | reason: `max_rounds`, `cost_limit`, `moderator_closed`, `manual`, `error` |
| `RoundStarted` / `RoundEnded` | round number |
| `TurnAssigned` | speaker, strategy name, rationale (if any) |
| `MessagePosted` | `Message` |
| `TurnSkipped` | reason: `human_timeout`, `passed`, `error` |
| `HandRaised` / `HandLowered` | participant |
| `ModeratorIntervened` | out-of-turn moderator message |
| `LLMCallCompleted` | model, purpose (`speech`, `summary`, `reflection`, `judge`), tokens, cost, ms |
| `MemoryWritten` | agent, layer, memory item id |
| `ReflectionProposed` | agent, proposed `StanceUpdate`s, `shadow` flag |
| `PersonaUpdated` | agent, version of the new state, applied changes, what it was triggered by |
| `PersonaUpdateRejected` / `Clamped` | agent, reason, original and modified value |
| `ApprovalRequested` / `ApprovalDecided` | for APPROVED mode |
| `ConsistencyViolation` | agent, judge score, rationale, whether regeneration happened |
| `Error` | component, message |

`SessionCreated` contains a full snapshot of the config, so an old session can be replayed exactly even if the template has been modified in the meantime.

---

## 7. Database Schema

```
agent_templates(id PK, config JSON, created_at, updated_at)
session_templates(id PK, config JSON, created_at, updated_at)

sessions(id PK, title, status, config JSON, created_at, ended_at)
   status: created | running | paused | ended | interrupted

events(id PK autoincrement, session_id FK, seq INT, type TEXT, payload JSON, created_at)
   UNIQUE(session_id, seq), INDEX(session_id, type)

-- projections
messages(id PK, session_id, seq, speaker_id, kind, content, reply_to, meta JSON, created_at)
messages_fts   -- FTS5 virtual table (content), content=messages

persona_versions(agent_id, session_id, version, state JSON, cause_seq, created_at)
   PK(agent_id, session_id, version)

memory_items(id PK autoincrement, agent_id, session_id NULL, layer, content,
             source_seq NULL, meta JSON, created_at)
   layer: episodic | long_term
memory_vec     -- vec0(embedding float[DIM]), rowid = memory_items.id
memory_fts     -- FTS5(content), content=memory_items

pending_approvals(id PK, session_id, agent_id, proposal JSON, status, created_at, decided_at)

app_meta(key PK, value)   -- e.g. embedding_model, embedding_dim
```

**Notes**

- Memory is queried filtered by `agent_id`: an agent only sees its own memories.
- Long-term memory has `agent_id` = the agent template id, so it spans sessions. Episodic memory is session-scoped.
- `app_meta` stores the embedding model name and dimension. If the config names a different model, the hub errors at startup, and the `thinktank reembed` command recomputes the vectors.
- FTS5 tokenizer: `unicode61`. For Hungarian text it is worth trying `remove_diacritics` with both values; in Hungarian diacritics carry meaning (kör/kór), so do **not** remove them by default. No stemming; the vector branch compensates for that.
- Pragmas at connection time: `journal_mode=WAL`, `foreign_keys=ON`, `busy_timeout=5000`, then load sqlite-vec.

---

## 8. Main Interfaces

### 8.1 Participant

```python
class TurnContext(BaseModel):
    session_id: UUID
    round: int
    turn_instruction: str | None = None   # e.g. a moderator request

class Participant(Protocol):
    id: str
    kind: Literal["ai", "human", "remote"]
    display_name: str

    async def speak(self, ctx: TurnContext) -> Message | None: ...   # None = pass / timeout
    async def observe(self, msg: Message) -> None: ...                # receives every message
    async def on_session_end(self) -> None: ...                       # e.g. long-term takeaways
```

- **AIAgent**: `speak()` → ContextBuilder → LLMClient → (optional consistency check) → Message. `observe()` → working memory is updated; when the `summarize_every` threshold is reached, it produces an episodic summary.
- **HumanParticipant**: `speak()` awaits an `asyncio.Future` released by WebSocket input; after `human_timeout_s` it returns `None` and a `TurnSkipped(human_timeout)` event is emitted. `observe()` does nothing (the UI gets the messages from the event bus).
- **RemoteAgent** (plugin, end of v1): an HTTP POST to an external endpoint with the context; the response is the Message. This lets an AG2, LangGraph, or any other agent sit at the table. Protocol: `POST {url}/speak` → `{"content": "..."}`, `POST {url}/observe`.

### 8.2 TurnStrategy

```python
class TurnStrategy(ABC):
    name: ClassVar[str]
    Params: ClassVar[type[BaseModel]] = EmptyParams   # the UI generates a form from this

    def __init__(self, params: BaseModel) -> None: ...

    @abstractmethod
    async def next_speaker(self, state: SessionState) -> str | None: ...

    async def on_event(self, event: Event) -> None:  # optional
        pass
```

- **RoundRobin** (built-in): a fixed order from the participant list, no moderator; parameter: `shuffle_each_round: bool`.
- **HandRaisePriority** (wrapper): wraps any strategy; if someone has raised their hand, they speak next; otherwise the inner strategy decides. This makes hand-raising strategy-independent.
- Later: **ModeratorPicks** (the moderator's LLM picks with structured output), **Bidding** (every AI agent gives a 0–1 score with a cheap call, the highest speaks; on a tie, whoever spoke longest ago).

The strategy defines the concept of a "round": the `SessionState.round_complete` flag is set by the strategy. With round robin this is trivial; with bidding, e.g., "N messages = one round".

### 8.3 MemoryBackend

```python
class MemoryBackend(ABC):
    @abstractmethod
    async def add(self, agent_id: str, layer: Layer, content: str,
                  session_id: UUID | None, source_seq: int | None, meta: dict) -> int: ...

    @abstractmethod
    async def search(self, agent_id: str, query: str, k: int,
                     layers: set[Layer], session_id: UUID | None = None) -> list[MemoryHit]: ...
```

### 8.4 DriftPolicy

```python
class DriftPolicy(ABC):
    name: ClassVar[str]

    @abstractmethod
    def evaluate(self, current: PersonaState, proposal: ReflectionResult,
                 config: DriftConfig) -> PolicyDecision: ...
    # PolicyDecision: apply(new_state) | reject(reason) | clamp(new_state, notes) | needs_approval
```

### 8.5 LLMClient

```python
class LLMClient(Protocol):
    async def complete(self, *, model: str, messages: list[ChatMessage],
                       purpose: Purpose, temperature: float, max_tokens: int,
                       response_model: type[BaseModel] | None = None) -> LLMResult: ...
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

The LiteLLM implementation is responsible for: tenacity retry (rate limit, 5xx, timeout), a per-provider `asyncio.Semaphore`, cost accounting (`litellm.completion_cost`), and emitting the `LLMCallCompleted` event. For structured output, try native `response_format` first; if the model does not support it: JSON extraction + Pydantic validation + one retry with the error message.

---

## 9. SessionEngine

### 9.1 Main Loop

```python
async def run(self) -> None:
    await self.emit(SessionStarted())
    await self.moderator_open()                       # if there is a moderator: opening framing
    while not await self.should_stop():
        await self.pause_gate.wait()                  # pause/resume
        speaker_id = await self.strategy.next_speaker(self.state)
        if speaker_id is None:
            break
        await self.emit(TurnAssigned(speaker_id=speaker_id, strategy=self.strategy.name))
        msg = await self.participants[speaker_id].speak(self.turn_context(speaker_id))
        if msg is None:
            await self.emit(TurnSkipped(speaker_id=speaker_id, reason=...))
        else:
            await self.post(msg)                      # MessagePosted + observe() for everyone
        if self.state.round_complete:
            await self.end_round()                    # RoundEnded, reflection, moderator summary
    await self.moderator_close()
    await self.finish()                               # on_session_end() for everyone, SessionEnded
```

### 9.2 Details

- **Fanning out observe():** `asyncio.gather` over all participants, and the next round starts only after that. Simple and deterministic. If embedding becomes the bottleneck, v2 could switch to per-agent queues, provided an agent's own queue drains before its own `speak()`.
- **Moderator:** a role, not a separate type. Speaks at the opening, at round ends (optionally, `moderator_every_rounds`), and at the close. A human moderator can insert a `ModeratorIntervened` message at any time; it enters every agent's context before the next round.
- **Stop conditions:** `max_rounds`, `max_messages`, `max_cost_usd` (sum of the `LLMCallCompleted` events), `max_duration_s`, moderator close, manual stop.
- **Error handling:** if an agent's LLM call still fails after the retries, an `Error` + `TurnSkipped(error)` event, and the debate continues. 3 consecutive errors from the same agent: the agent drops out of the rest of the session (`ParticipantDisabled`).
- **Recovery after a crash:** at startup, `running` sessions become `interrupted`. On resume, the `SessionState` is rebuilt by replaying the events (the same projection code as in the live run), and the loop starts from the next turn.

---

## 10. Context Assembly (AI Agent)

The messages of one call, in this order:

1. **System**
   - The persona **core**, rendered (again on every call).
   - Behavior rules: speak in your own name, respond concretely to others' arguments by name, do not speak in someone else's name, keep within the length limit, do not repeat your own earlier arguments.
   - The list of participants with a one-line description each.
   - The topic and the debate questions.
   - The persona's **current state** (stances + confidence, attitudes).
2. **Memories** (as a system supplement or a separate block): episodic and long-term items retrieved by hybrid search.
3. **Own episodic summary** of the earlier segment.
4. **Working memory:** the last `working_window` messages.
   - Others' messages in the `user` role, prefixed with `[Name]: ...`; own earlier messages in the `assistant` role. Consecutive `user` messages must be merged into a single message, because several providers do not accept consecutive same-role messages.
5. **Turn instruction:** "You are next. …" + any moderator request.

**Token budget:** per-section budgets (e.g., system 1500, memories 1000, summary 600, working memory takes the rest). When over, the cut order is: memories → oldest working memory items → summary. The core and the state are never cut. Use `litellm.token_counter` for token counting.

**Language:** the language of the prompt templates is configurable. Worth comparing: English instructions + "respond in Hungarian" vs. a fully Hungarian prompt (persona stability can differ per model).

---

## 11. Memory

| Layer | Content | Origin | Scope |
|---|---|---|---|
| Working memory | last N messages verbatim | from the `messages` projection, no separate storage | session |
| Episodic | the agent's **own-perspective** summary: what it heard, what convinced it, what it disagrees with | every `summarize_every` messages, may use a cheaper model | session |
| Long-term | takeaways, important insights, connections to other agents | `on_session_end()` at the end of the session | agent template, across sessions |

**Hybrid search:** the query is the topic + the last 1–2 messages. Vector top-k (sqlite-vec) and FTS5 BM25 top-k, merged with **Reciprocal Rank Fusion** (`score = Σ 1/(60 + rank)`), optional recency weighting. Always filter by `agent_id`.

**Embedding:** `EmbeddingProvider` interface, two built-in implementations:
- `sentence_transformers`: local, e.g., `paraphrase-multilingual-MiniLM-L12-v2` (fast) or `BAAI/bge-m3` (better, larger). MPS acceleration on Mac.
- `litellm`: Ollama or API-based embedding.

For Hungarian-language debates, only multilingual models are worth using.

---

## 12. Persona Management

### 12.1 Stability

- The core appears at the top of the system prompt on every call.
- The reflection schema contains **only `PersonaState` fields**, so the core is not modifiable even at the type level.
- **Consistency check** (toggleable per agent): a judge model receives the core and the candidate answer, and returns a 1–5 score and a rationale. Below the threshold, one regeneration with the judge's feedback; a `ConsistencyViolation` event in both cases. Expensive, so off by default.

### 12.2 Reflection

At the end of every `reflection_every_rounds` rounds, for every agent whose mode is not LOCKED (or where `shadow_reflection` is enabled):

```python
class StanceUpdate(BaseModel):
    question_id: str
    new_position: str
    new_confidence: float = Field(ge=0, le=1)
    influenced_by: list[str]          # participant ids
    reason: str

class AttitudeUpdate(BaseModel):
    participant_id: str
    new_value: float = Field(ge=-1, le=1)
    reason: str

class ReflectionResult(BaseModel):
    stance_updates: list[StanceUpdate] = []
    attitude_updates: list[AttitudeUpdate] = []
    mood: str | None = None
```

The reflection prompt explicitly allows, even expects, "no change" as the answer when nothing has convinced the agent. This reduces artificial convergence.

### 12.3 Policy Application

| Mode | Behavior |
|---|---|
| LOCKED | reject; in shadow mode `ReflectionProposed(shadow=True)` is logged |
| BOUNDED | `confidence` and `attitude` changes are clamped to the maximum; at most `max_stance_changes_per_round` stances may change; `Clamped` event |
| APPROVED | `pending_approvals` + `ApprovalRequested`; approve/reject in the UI; until then the old state is in effect, the debate does not stop |
| FREE | apply |

Every applied change is a new `persona_versions` row + a `PersonaUpdated` event with the `influenced_by` field. From this the agent inspector's drift timeline can be drawn, and this is what the influence graph (who moved whose opinion) is built from.

---

## 13. Plugin System

**Entry point groups**

| Group | Base class |
|---|---|
| `thinktank.turn_strategies` | `TurnStrategy` |
| `thinktank.participants` | `Participant` factory (AI agent types, RemoteAgent) |
| `thinktank.memory_backends` | `MemoryBackend` |
| `thinktank.drift_policies` | `DriftPolicy` |
| `thinktank.embedding_providers` | `EmbeddingProvider` |

The built-in implementations are also registered in their own `pyproject.toml`, so they are loaded the same way as external ones:

```toml
[project.entry-points."thinktank.turn_strategies"]
round_robin = "thinktank.strategies.round_robin:RoundRobin"
```

**Registry:** at startup, load `importlib.metadata.entry_points(group=...)` and verify (`issubclass`, has `name` and `Params`); a bad plugin produces a warning in the log, but the hub still starts. `GET /api/plugins` returns the list of plugins with `Params.model_json_schema()`, and the UI generates a configuration form from this. A new strategy thus appears without any UI changes.

---

## 14. API

### 14.1 REST

```
GET    /api/plugins                              plugins + parameter schemas

GET    /api/agents                               agent templates
POST   /api/agents
GET    /api/agents/{id}
PUT    /api/agents/{id}
DELETE /api/agents/{id}
GET    /api/agents/{id}/memory?q=                browsing long-term memory

GET    /api/session-templates                    (same CRUD)

POST   /api/sessions                             from a template or an inline config
GET    /api/sessions
GET    /api/sessions/{id}
POST   /api/sessions/{id}/start | pause | resume | stop
GET    /api/sessions/{id}/events?after_seq=&types=
GET    /api/sessions/{id}/agents/{aid}/state     current persona state
GET    /api/sessions/{id}/agents/{aid}/history   persona_versions
GET    /api/sessions/{id}/agents/{aid}/memory?q=
GET    /api/sessions/{id}/approvals
POST   /api/sessions/{id}/approvals/{pid}        {"decision": "approve" | "reject"}
GET    /api/sessions/{id}/export?format=jsonl|csv
```

### 14.2 WebSocket

`/ws/sessions/{id}?participant=<id>&token=<t>&after_seq=<n>`

- On connect, the server sends all events after `after_seq` from the database, then continues live. Reconnection is thus lossless.
- Server → client: `{"type": "event", "seq": 42, "event": {...}}`, plus `{"type": "your_turn", "deadline": "..."}` for a human participant.
- Client → server: `{"type": "say", "content": "..."}`, `{"type": "raise_hand"}`, `{"type": "lower_hand"}`, `{"type": "intervene", "content": "..."}` (moderator only).

### 14.3 Access

v1: the hub binds to `127.0.0.1`. A token generated per human participant is placed in the URL. If it also has to be reachable over a LAN, a simple admin token specified in `.env` is required for the configuration endpoints.

---

## 15. Web UI

| View | Content |
|---|---|
| Boardroom | list of sessions with state and cost; new session |
| Persona Templates | agent templates; form: model, core, initial state, drift mode, memory, consistency check |
| Session builder | topic, debate questions, participant selection, moderator, strategy (parameter form generated from the schema), stop conditions; save as a template |
| Live table | transcript; sidebar with the participants (who is speaking, who is "thinking", whose hand is raised); round and cost meter; pause/resume/stop; human input field and raise-hand button; approval panel for APPROVED mode |
| Agent inspector | core, current state, stance timeline per question (Chart.js), memory search, consistency events |
| Replay and export | a closed session, step-through per event; JSONL/CSV export |

Implementation: Jinja2 templates, HTMX for partial updates, `htmx-ext-ws` for live events; the server sends HTML snippets from the events.

---

## 16. Example Session Template

```yaml
title: "Can an LLM be a co-author?"
topic: >
  Generative AI is taking an ever-larger role in scientific paper writing.
  Debate the questions of authorship, responsibility, and contribution.
language: en
questions:
  - id: q_authorship
    text: "Can an LLM be listed as an author on a scientific paper?"
  - id: q_disclosure
    text: "Should detailed disclosure of AI usage be mandatory?"
participants:
  - agent: skeptic_methodologist
  - agent: pragmatic_editor
  - agent: ai_optimist
  - human: "Sam"
moderator: ai_moderator
strategy:
  name: round_robin
  params: { shuffle_each_round: false }
stop:
  max_rounds: 5
  max_cost_usd: 1.5
reflection_every_rounds: 1
```

---

## 17. Testing

- **Unit:** strategies (deterministic order, hand-raising priority), drift policies (clamping, rejection), RRF fusion, context assembly with the token budget, prompt rendering (snapshot tests).
- **Integration:** a full session with `FakeLLM` (scripted responses), on a temporary SQLite database; checks: event order, projections, stop conditions.
- **Replay test:** a `SessionState` rebuilt from the events of a completed session matches the final state of the run.
- **WebSocket:** `httpx` + FastAPI TestClient; reconnection with `after_seq` loses no events.
- **Live:** `pytest -m live`, real API, skipped by default; a short two-agent, two-round debate.
- CI: ruff, pyright strict, pytest (no live).

---

## 18. Milestones

| # | Content | Done when |
|---|---|---|
| M0 | Repo, uv, ruff, pyright, pytest; Settings; DB + Alembic; pragmas; **loading sqlite-vec on Mac and PC** | tests are green and sqlite-vec loads under aiosqlite on both machines |
| M1 | Domain model, events, SessionEngine, RoundRobin, AIAgent, LiteLLM client, FakeLLM, CLI `thinktank run template.yaml` | a three-agent debate runs from the command line, the events are in the database, the replay test is green |
| M2 | FastAPI hub, WebSocket, live table view, HumanParticipant, pause/resume/stop, cost limit | a debate can be started from the browser, a human contributes, the transcript is intact after a reload |
| M3 | Memory: episodic summaries, long-term takeaways, sqlite-vec + FTS5 + RRF, EmbeddingProvider | in a second session the agent refers to a takeaway from the previous session |
| M4 | Persona: core/state, debate questions, reflection, LOCKED + FREE + shadow, persona_versions, agent inspector with a timeline | a visible and justified stance change in FREE mode; in LOCKED mode the state does not change and the shadow proposals are logged |
| M5 | Plugin registry with entry points, schema-generated forms, HandRaisePriority, a second strategy (ModeratorPicks or Bidding) | a strategy from a separately installed package appears in the UI and is usable |
| M6 | BOUNDED and APPROVED mode, approval panel, consistency check, export, replay view | every drift mode works; the JSONL export is ready for analysis |
| M7 (optional) | RemoteAgent, token streaming to the UI, tool-using agents (e.g., literature search) | |

After M1, the system can already be used for research experiments from the command line, without the UI.

---

## 19. Risks and Open Questions

- **Loading sqlite-vec:** the macOS system Python and some Python builds do not allow `enable_load_extension`. The first task of M0 is to verify this with the uv-installed Python, under aiosqlite, on both machines.
- **Convergence and sycophancy:** agents tend to agree quickly. Countermeasures: different models for the agents, a strong and concrete core, the "no change" option in the reflection prompt, a devil's advocate role, and BOUNDED mode.
- **Cost:** N agents × rounds × (speech + summary + reflection + optional judge). The cost limit is a required default; a cheaper model can be configured for the side calls (summary, reflection, judge).
- **Multi-participant conversation in chat format:** the "others = user, self = assistant" mapping works, but it is worth comparing, per model, with the setup where the whole transcript arrives in a single user message.
- **Fixed debate questions:** in v1 the questions must be given at the start of the session. It is an open question whether the moderator or the agents can add a new question during a run later.
- **Prompt language:** Hungarian or English instructions with Hungarian output; the effect on persona stability must be measured.
