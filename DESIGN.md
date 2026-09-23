# Roundtable – tervezési dokumentum (v1)

Többszereplős vitarendszer, ahol AI-agentek és emberek egy virtuális asztalnál beszélgetnek egy témáról. Mindenki hall mindenkit, az agentek memóriával rendelkeznek, a perszónájuk rögzíthető vagy szabályozottan fejlődhet. A rendszert egy webes hub konfigurálja és követi.

---

## 1. Célok és nem-célok

**Célok (v1)**

- AI-agentek és emberi résztvevők egyenrangúan, azonos interfészen keresztül vesznek részt.
- A moderátor lehet ember vagy AI.
- A szólási sorrend cserélhető pluginként; alapértelmezés: round robin.
- Agentenként háromrétegű memória (munka, epizodikus, hosszú távú).
- A perszóna rögzített magból és változtatható állapotból áll; a változás módja agentenként szabályozható.
- Webes hub: konfiguráció, élő követés, emberi hozzászólás, agent-inspektor.
- Minden történés eseményként tárolódik: visszajátszható, exportálható kutatási elemzéshez.
- Egyetlen gépen fut (PC vagy Mac), SQLite-tal.

**Nem-célok (v1)**

- Postgres-támogatás, több felhasználós hozzáférés-kezelés, internetre kitett telepítés.
- Token-szintű streaming a UI-ba (v2).
- Hang- vagy videóalapú részvétel.

---

## 2. Technológiai stack

| Terület | Választás | Megjegyzés |
|---|---|---|
| Nyelv | Python 3.12+ | |
| Csomag- és környezetkezelés | uv | a Pythont is uv telepítse (lásd 14. kockázatok) |
| Web | FastAPI + uvicorn | REST + WebSocket |
| Sémák | Pydantic v2, pydantic-settings | konfig, események, API |
| Adatbázis | SQLite, SQLAlchemy 2.0 async + aiosqlite | WAL mód |
| Migráció | Alembic, `render_as_batch=True` | SQLite oszlopmódosításhoz kell |
| Vektoros keresés | sqlite-vec | `vec0` virtuális tábla |
| Teljes szöveges keresés | SQLite FTS5 | hibrid keresés a vektorossal |
| LLM | LiteLLM (`acompletion`, `aembedding`) | Claude, GPT, Gemini, Ollama egy felületen |
| Embedding | sentence-transformers vagy Ollama | **többnyelvű** modell a magyar miatt |
| UI | Jinja2 + HTMX (+ ws extension) | build lépés nélkül |
| Grafikon | Chart.js, statikus fájlként | drift-idővonal |
| Retry | tenacity | |
| Naplózás | structlog | JSON-napló |
| Minőség | ruff, pyright strict, pytest + pytest-asyncio, pre-commit | |

---

## 3. Architektúra

```
┌──────────────────── Web UI (Jinja2 + HTMX) ────────────────────┐
│   REST: konfiguráció, vezérlés      WebSocket: élő események   │
└───────────────────────────────┬────────────────────────────────┘
                           FastAPI Hub
     ┌──────────────┬───────────┴──────┬──────────────────┐
 SessionManager  PluginRegistry     EventBus         Repositories
 (futó sessionök) (entry points)   (in-process)      (SQLite)
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

**Alapelvek**

1. **Event sourcing.** Az `events` tábla az igazság forrása. A `messages`, `persona_versions` és hasonló táblák ebből képzett projekciók. Az élő UI, az újracsatlakozás, a visszajátszás és az export mind az eseményfolyamra épül.
2. **Egy interfész minden résztvevőre.** A motor nem tudja, hogy ember vagy AI beszél.
3. **Minden cserélhető rész plugin**, ugyanazzal a regisztrációs mechanizmussal, a beépítetteket is beleértve.
4. **Az LLM-hívás injektált függőség** (`LLMClient` protokoll), így a mag teljes egészében tesztelhető `FakeLLM`-mel, API-hívás nélkül.

---

## 4. Könyvtárszerkezet

```
roundtable/
├── pyproject.toml
├── alembic.ini
├── .env.example
├── migrations/
├── templates/                   # mentett session- és agent-sablonok (YAML)
│   ├── agents/
│   └── sessions/
├── src/roundtable/
│   ├── config.py                # Settings (pydantic-settings)
│   ├── cli.py                   # roundtable run / serve / export / reembed
│   ├── domain/
│   │   ├── models.py            # AgentConfig, PersonaCore, PersonaState, SessionConfig, Message
│   │   └── events.py            # eseménytípusok
│   ├── core/
│   │   ├── engine.py            # SessionEngine
│   │   ├── manager.py           # SessionManager (több session, életciklus)
│   │   ├── state.py             # SessionState + projekció eseményekből
│   │   ├── context.py           # ContextBuilder, tokenkeret
│   │   └── bus.py               # EventBus (pub/sub, WS felé)
│   ├── participants/
│   │   ├── base.py              # Participant protokoll
│   │   ├── ai_agent.py
│   │   ├── human.py
│   │   └── remote.py            # HTTP-n csatlakozó külső agent (v1 vége / v2)
│   ├── strategies/
│   │   ├── base.py              # TurnStrategy ABC
│   │   ├── round_robin.py
│   │   └── hand_raise.py        # wrapper: jelentkezés elsőbbsége bármely stratégián
│   ├── memory/
│   │   ├── base.py              # MemoryBackend ABC
│   │   ├── sqlite_memory.py     # sqlite-vec + FTS5, RRF-fúzió
│   │   ├── embeddings.py        # EmbeddingProvider
│   │   └── summarizer.py        # epizodikus összefoglalók
│   ├── persona/
│   │   ├── manager.py
│   │   ├── policies.py          # LOCKED, BOUNDED, APPROVED, FREE
│   │   ├── reflection.py
│   │   └── consistency.py
│   ├── llm/
│   │   ├── client.py            # LiteLLM wrapper: retry, timeout, költség, szemafor
│   │   ├── fake.py              # FakeLLM tesztekhez
│   │   └── prompts/             # Jinja2 promptsablonok
│   ├── plugins/
│   │   └── registry.py
│   ├── storage/
│   │   ├── db.py                # engine, pragmák, sqlite-vec betöltés
│   │   ├── tables.py            # ORM-modellek
│   │   └── repositories.py
│   ├── api/
│   │   ├── app.py
│   │   ├── routes_library.py    # agent- és session-sablonok
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
    └── live/                    # valódi API-hívás, alapból kihagyva
```

---

## 5. Domain modell

### 5.1 Perszóna

```python
class PersonaCore(BaseModel, frozen=True):
    """Soha nem változik a session alatt."""
    name: str
    role: str                          # pl. "szkeptikus módszertanász"
    expertise: list[str]
    values: list[str]
    communication_style: str           # pl. "tömör, kérdez vissza, példákat hoz"
    temperament: str
    background: str = ""
    boundaries: list[str] = []         # amit sosem tesz/mond

class Stance(BaseModel):
    position: str                      # szöveges álláspont
    confidence: float = Field(ge=0, le=1)

class PersonaState(BaseModel):
    """A drift policy szerint változhat."""
    stances: dict[str, Stance] = {}    # kulcs: a session egy vitakérdésének id-je
    attitudes: dict[str, float] = {}   # résztvevő id -> bizalom (-1..1)
    mood: str = "semleges"
```

### 5.2 Drift policy

```python
class DriftMode(StrEnum):
    LOCKED = "locked"      # állapot sem változhat
    BOUNDED = "bounded"    # változhat, lépésköz-korláttal
    APPROVED = "approved"  # javaslat → ember/moderátor jóváhagyja
    FREE = "free"          # szabad; a mag ekkor is védett

class DriftConfig(BaseModel):
    mode: DriftMode = DriftMode.LOCKED
    max_confidence_delta: float = 0.2     # BOUNDED: körönként
    max_attitude_delta: float = 0.3
    max_stance_changes_per_round: int = 1
    shadow_reflection: bool = False       # LOCKED mellett is lefut a reflexió, csak nem alkalmazzuk
```

A `shadow_reflection` kutatási célú: méri, mekkora „nyomás” éri a rögzített perszónát, anélkül hogy engedne neki.

### 5.3 Agent és session konfiguráció

```python
class MemoryConfig(BaseModel):
    working_window: int = 12           # utolsó N üzenet szó szerint
    summarize_every: int = 8           # ennyi új üzenetenként epizodikus összefoglaló
    retrieval_k: int = 5
    long_term: bool = True             # sessionök közti memória

class AgentConfig(BaseModel):
    id: str                            # sablon-azonosító = hosszú távú identitás
    type: str = "llm"                  # participant plugin neve
    model: str                         # LiteLLM modellstring
    temperature: float = 0.8
    max_tokens: int = 600
    persona: PersonaCore
    initial_state: PersonaState = PersonaState()
    drift: DriftConfig = DriftConfig()
    memory: MemoryConfig = MemoryConfig()
    consistency_check: bool = False
    carry_over_state: bool = False     # perszóna-állapot átvitele a következő sessionbe

class DebateQuestion(BaseModel):
    id: str                            # pl. "q_authorship"
    text: str                          # pl. "Lehet-e egy LLM társszerző?"

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
    topic: str                         # a moderátor nyitó felvetése / kutatási terület
    questions: list[DebateQuestion]    # ezek mentén mérjük az álláspontokat
    participants: list[ParticipantRef] # agent-sablon id vagy ember (név)
    moderator: str | None              # résztvevő id, vagy None
    strategy: StrategyRef = StrategyRef()
    stop: StopConditions = StopConditions()
    reflection_every_rounds: int = 1
    language: str = "hu"
```

**Miért kellenek a `questions`?** Ha az álláspontok előre definiált kérdésekhez kötődnek, a drift mérhető és összehasonlítható: minden kérdésre, minden agentre, minden körben van egy `position` és egy `confidence`. Szabad szöveges „véleményváltozásból” ez nem nyerhető ki megbízhatóan.

### 5.4 Üzenet

```python
class Message(BaseModel):
    id: UUID
    session_id: UUID
    seq: int
    speaker_id: str
    kind: Literal["speech", "moderator", "system"]
    content: str
    reply_to: UUID | None = None
    meta: dict[str, Any] = {}          # modell, tokenek, költség, késleltetés
```

---

## 6. Események

Minden esemény: `session_id`, `seq` (sessionön belül szigorúan növekvő, az alkalmazás osztja ki), `type`, `payload` (JSON), `created_at` (UTC).

| Esemény | Payload lényege |
|---|---|
| `SessionCreated` | teljes `SessionConfig` pillanatképe |
| `SessionStarted` / `Paused` / `Resumed` | |
| `SessionEnded` | ok: `max_rounds`, `cost_limit`, `moderator_closed`, `manual`, `error` |
| `RoundStarted` / `RoundEnded` | körszám |
| `TurnAssigned` | beszélő, stratégia neve, indoklás (ha van) |
| `MessagePosted` | `Message` |
| `TurnSkipped` | ok: `human_timeout`, `passed`, `error` |
| `HandRaised` / `HandLowered` | résztvevő |
| `ModeratorIntervened` | soron kívüli moderátori üzenet |
| `LLMCallCompleted` | modell, célja (`speech`, `summary`, `reflection`, `judge`), tokenek, költség, ms |
| `MemoryWritten` | agent, réteg, memória-elem id |
| `ReflectionProposed` | agent, javasolt `StanceUpdate`-ek, `shadow` jelző |
| `PersonaUpdated` | agent, új állapot verziószáma, alkalmazott változások, ki hatására |
| `PersonaUpdateRejected` / `Clamped` | agent, ok, eredeti és módosított érték |
| `ApprovalRequested` / `ApprovalDecided` | APPROVED módhoz |
| `ConsistencyViolation` | agent, bírói pontszám, indoklás, újragenerálás történt-e |
| `Error` | komponens, üzenet |

A `SessionCreated` a teljes konfig pillanatképét tartalmazza, így egy régi session akkor is pontosan visszajátszható, ha közben a sablont módosítottad.

---

## 7. Adatbázisséma

```
agent_templates(id PK, config JSON, created_at, updated_at)
session_templates(id PK, config JSON, created_at, updated_at)

sessions(id PK, title, status, config JSON, created_at, ended_at)
   status: created | running | paused | ended | interrupted

events(id PK autoincrement, session_id FK, seq INT, type TEXT, payload JSON, created_at)
   UNIQUE(session_id, seq), INDEX(session_id, type)

-- projekciók
messages(id PK, session_id, seq, speaker_id, kind, content, reply_to, meta JSON, created_at)
messages_fts   -- FTS5 virtuális tábla (content), content=messages

persona_versions(agent_id, session_id, version, state JSON, cause_seq, created_at)
   PK(agent_id, session_id, version)

memory_items(id PK autoincrement, agent_id, session_id NULL, layer, content,
             source_seq NULL, meta JSON, created_at)
   layer: episodic | long_term
memory_vec     -- vec0(embedding float[DIM]), rowid = memory_items.id
memory_fts     -- FTS5(content), content=memory_items

pending_approvals(id PK, session_id, agent_id, proposal JSON, status, created_at, decided_at)

app_meta(key PK, value)   -- pl. embedding_model, embedding_dim
```

**Megjegyzések**

- A memória `agent_id` szerint szűrve kerül lekérdezésre: egy agent csak a saját emlékeit látja.
- A hosszú távú memória `agent_id` = az agent-sablon id-je, így sessionökön átível. Az epizodikus memória session-szintű.
- Az `app_meta` tárolja az embedding-modell nevét és dimenzióját. Ha a konfigban más modell szerepel, a hub induláskor hibát jelez, és a `roundtable reembed` parancs újraszámolja a vektorokat.
- FTS5 tokenizer: `unicode61`. A `remove_diacritics` beállítást magyar szövegen érdemes kipróbálni mindkét értékkel; a magyarban az ékezet jelentést hordoz (kör/kór), ezért alapból **ne** távolítsd el. Szótövezés nincs, ezt a vektoros ág kompenzálja.
- Pragmák kapcsolódáskor: `journal_mode=WAL`, `foreign_keys=ON`, `busy_timeout=5000`, majd a sqlite-vec betöltése.

---

## 8. Fő interfészek

### 8.1 Participant

```python
class TurnContext(BaseModel):
    session_id: UUID
    round: int
    turn_instruction: str | None = None   # pl. moderátori felkérés

class Participant(Protocol):
    id: str
    kind: Literal["ai", "human", "remote"]
    display_name: str

    async def speak(self, ctx: TurnContext) -> Message | None: ...   # None = passzol / timeout
    async def observe(self, msg: Message) -> None: ...                # minden üzenetet megkap
    async def on_session_end(self) -> None: ...                       # pl. hosszú távú tanulságok
```

- **AIAgent**: `speak()` → ContextBuilder → LLMClient → (opcionális konzisztencia-ellenőrzés) → Message. `observe()` → munkamemória frissül; ha elérte a `summarize_every` küszöböt, epizodikus összefoglalót készít.
- **HumanParticipant**: `speak()` egy `asyncio.Future`-re vár, amit a WebSocket-bemenet old fel; `human_timeout_s` után `None`, és `TurnSkipped(human_timeout)` esemény keletkezik. A `observe()` nem csinál semmit (a UI az eseménybuszról kapja az üzeneteket).
- **RemoteAgent** (plugin, v1 vége): HTTP POST egy külső végpontra a kontextussal, a válasz a Message. Így AG2-, LangGraph- vagy bármilyen más agent is leülhet az asztalhoz. Protokoll: `POST {url}/speak` → `{"content": "..."}`, `POST {url}/observe`.

### 8.2 TurnStrategy

```python
class TurnStrategy(ABC):
    name: ClassVar[str]
    Params: ClassVar[type[BaseModel]] = EmptyParams   # a UI ebből generál űrlapot

    def __init__(self, params: BaseModel) -> None: ...

    @abstractmethod
    async def next_speaker(self, state: SessionState) -> str | None: ...

    async def on_event(self, event: Event) -> None:  # opcionális
        pass
```

- **RoundRobin** (beépített): fix sorrend a résztvevőlistából, moderátor nélkül; paraméter: `shuffle_each_round: bool`.
- **HandRaisePriority** (wrapper): bármely stratégiát becsomagol; ha van jelentkező, ő következik, egyébként a belső stratégia dönt. Így a jelentkezés nem stratégiafüggő.
- Később: **ModeratorPicks** (a moderátor LLM strukturált kimenettel választ), **Bidding** (minden AI-agent egy olcsó hívással 0–1 pontszámot ad, a legmagasabb szól; holtverseny esetén az, aki régebben beszélt).

A „kör” fogalmát a stratégia definiálja: a `SessionState.round_complete` jelzőt a stratégia állítja. Round robinnál ez triviális; bidding esetén pl. „N üzenet = egy kör”.

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

A LiteLLM-implementáció feladata: tenacity-retry (rate limit, 5xx, timeout), szolgáltatónkénti `asyncio.Semaphore`, költségszámítás (`litellm.completion_cost`), `LLMCallCompleted` esemény kibocsátása. Strukturált kimenetnél először natív `response_format`, ha a modell nem támogatja: JSON-kinyerés + Pydantic-validáció + egy újrapróbálás a hibaüzenettel.

---

## 9. SessionEngine

### 9.1 Fő ciklus

```python
async def run(self) -> None:
    await self.emit(SessionStarted())
    await self.moderator_open()                       # ha van moderátor: nyitó felvetés
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
            await self.post(msg)                      # MessagePosted + observe() mindenkinek
        if self.state.round_complete:
            await self.end_round()                    # RoundEnded, reflexió, moderátori összegzés
    await self.moderator_close()
    await self.finish()                               # on_session_end() mindenkinek, SessionEnded
```

### 9.2 Részletek

- **observe() szétterítése:** `asyncio.gather` az összes résztvevőre, és a következő kör csak ezután indul. Egyszerű és determinisztikus. Ha az embedding lassúvá teszi, v2-ben agentenkénti sorra lehet váltani, azzal a feltétellel, hogy egy agent saját `speak()`-je előtt a saját sora kiürül.
- **Moderátor:** szerep, nem külön típus. Megszólal nyitáskor, körvégén (opcionálisan, `moderator_every_rounds`), záráskor. Emberi moderátor bármikor beszúrhat `ModeratorIntervened` üzenetet; ez a következő kör előtt minden agent kontextusába bekerül.
- **Leállási feltételek:** `max_rounds`, `max_messages`, `max_cost_usd` (az `LLMCallCompleted` események összege), `max_duration_s`, moderátori zárás, kézi leállítás.
- **Hibakezelés:** ha egy agent LLM-hívása a retry-k után is elbukik, `Error` + `TurnSkipped(error)` esemény, és a vita folytatódik. Egymás után 3 hiba ugyanattól az agenttől: az agent kimarad a session hátralévő részéből (`ParticipantDisabled`).
- **Összeomlás utáni helyreállítás:** induláskor a `running` állapotú sessionök `interrupted` lesznek. Folytatáskor a `SessionState` az események visszajátszásából épül újra (ugyanaz a projekciós kód, mint az élő futásnál), és a ciklus a következő fordulótól indul.

---

## 10. Kontextus-összeállítás (AI-agent)

Egy hívás üzenetei, ebben a sorrendben:

1. **System**
   - A perszóna **mag** renderelve (minden hívásnál újra).
   - Viselkedési szabályok: saját nevedben beszélj, reagálj konkrétan mások érveire név szerint, ne beszélj más nevében, tartsd a hosszkeretet, ne ismételd a korábbi saját érveidet.
   - A résztvevők listája egy-egy soros leírással.
   - A téma és a vitakérdések.
   - A perszóna **aktuális állapota** (álláspontok + bizonyosság, attitűdök).
2. **Emlékek** (system-kiegészítésként vagy külön blokkban): hibrid kereséssel lekért epizodikus és hosszú távú elemek.
3. **Saját epizodikus összefoglaló** a korábbi szakaszról.
4. **Munkamemória:** az utolsó `working_window` üzenet.
   - Mások üzenetei `user` szerepben, `[Név]: ...` előtaggal; a saját korábbi üzenetei `assistant` szerepben. Egymást követő `user` üzeneteket egyetlen üzenetbe kell összevonni, mert több szolgáltató nem fogad el azonos szerepű egymás utáni üzeneteket.
5. **Fordulóutasítás:** „Te következel. …” + az esetleges moderátori felkérés.

**Tokenkeret:** szakaszonkénti keret (pl. system 1500, emlékek 1000, összefoglaló 600, munkamemória a maradék). Túllépéskor a vágási sorrend: emlékek → munkamemória legrégebbi elemei → összefoglaló. A mag és az állapot sosem vágható. Tokenszámláshoz `litellm.token_counter`.

**Nyelv:** a promptsablonok nyelve konfigurálható. Érdemes összehasonlítani: angol nyelvű utasítás + „válaszolj magyarul” vs. teljesen magyar prompt (a perszóna-stabilitás modellenként eltérhet).

---

## 11. Memória

| Réteg | Tartalom | Keletkezés | Hatókör |
|---|---|---|---|
| Munkamemória | utolsó N üzenet szó szerint | a `messages` projekcióból, nincs külön tárolás | session |
| Epizodikus | az agent **saját nézőpontú** összefoglalója: mit hallott, mi győzte meg, mivel nem ért egyet | `summarize_every` üzenetenként, olcsóbb modellel is mehet | session |
| Hosszú távú | tanulságok, fontos felismerések, kapcsolatok más agentekkel | session végén `on_session_end()` | agent-sablon, sessionökön át |

**Hibrid keresés:** a lekérdezés a téma + az utolsó 1–2 üzenet. Vektoros top-k (sqlite-vec) és FTS5 BM25 top-k, összefésülés **Reciprocal Rank Fusion**-nel (`score = Σ 1/(60 + rank)`), opcionális frissességi súllyal. Szűrés mindig `agent_id` szerint.

**Embedding:** `EmbeddingProvider` interfész, két beépített implementációval:
- `sentence_transformers`: lokális, pl. `paraphrase-multilingual-MiniLM-L12-v2` (gyors) vagy `BAAI/bge-m3` (jobb, nagyobb). Macen MPS-gyorsítással.
- `litellm`: Ollama vagy API-s embedding.

Magyar nyelvű vitákhoz csak többnyelvű modellt érdemes használni.

---

## 12. Perszóna-kezelés

### 12.1 Stabilitás

- A mag minden hívásnál a system prompt elején szerepel.
- A reflexió sémája **csak `PersonaState` mezőket** tartalmaz, így a mag típusszinten sem módosítható.
- **Konzisztencia-ellenőrzés** (agentenként kapcsolható): egy bíramodell megkapja a magot és a jelölt választ, 1–5 pontszámot és indoklást ad. Küszöb alatt egyszeri újragenerálás a bíró visszajelzésével; mindkét esetben `ConsistencyViolation` esemény. Költséges, ezért alapból kikapcsolt.

### 12.2 Reflexió

Minden `reflection_every_rounds` kör végén, minden olyan agentre, ahol a mód nem LOCKED (vagy ahol `shadow_reflection` be van kapcsolva):

```python
class StanceUpdate(BaseModel):
    question_id: str
    new_position: str
    new_confidence: float = Field(ge=0, le=1)
    influenced_by: list[str]          # résztvevő id-k
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

A reflexiós prompt kifejezetten megengedi, sőt elvárja, hogy „nincs változás” legyen a válasz, ha semmi nem győzte meg az agentet. Ez csökkenti a mesterséges konvergenciát.

### 12.3 Policy-alkalmazás

| Mód | Viselkedés |
|---|---|
| LOCKED | elutasít; shadow módban `ReflectionProposed(shadow=True)` naplózódik |
| BOUNDED | `confidence` és `attitude` változás levágva a maximumra; legfeljebb `max_stance_changes_per_round` álláspont változhat; `Clamped` esemény |
| APPROVED | `pending_approvals` + `ApprovalRequested`; a UI-ban jóváhagyás/elutasítás; addig a régi állapot érvényes, a vita nem áll meg |
| FREE | alkalmaz |

Minden alkalmazott változás új `persona_versions` sor + `PersonaUpdated` esemény, az `influenced_by` mezővel. Ebből rajzolható ki az agent-inspektor drift-idővonala, és ebből készül a hatásgráf (ki kinek a véleményét mozdította el).

---

## 13. Pluginrendszer

**Entry point csoportok**

| Csoport | Alaposztály |
|---|---|
| `roundtable.turn_strategies` | `TurnStrategy` |
| `roundtable.participants` | `Participant`-factory (AI-agenttípusok, RemoteAgent) |
| `roundtable.memory_backends` | `MemoryBackend` |
| `roundtable.drift_policies` | `DriftPolicy` |
| `roundtable.embedding_providers` | `EmbeddingProvider` |

A beépített implementációk is a saját `pyproject.toml`-ban vannak regisztrálva, így ugyanazon az úton töltődnek be, mint a külsők:

```toml
[project.entry-points."roundtable.turn_strategies"]
round_robin = "roundtable.strategies.round_robin:RoundRobin"
```

**Registry:** induláskor `importlib.metadata.entry_points(group=...)` betöltése, ellenőrzés (`issubclass`, van-e `name` és `Params`), hibás plugin esetén figyelmeztetés a naplóban, a hub ettől még elindul. A `GET /api/plugins` visszaadja a pluginek listáját a `Params.model_json_schema()`-val, a UI ebből generál konfigurációs űrlapot. Egy új stratégia így UI-módosítás nélkül megjelenik.

---

## 14. API

### 14.1 REST

```
GET    /api/plugins                              pluginek + paraméter-sémák

GET    /api/agents                               agent-sablonok
POST   /api/agents
GET    /api/agents/{id}
PUT    /api/agents/{id}
DELETE /api/agents/{id}
GET    /api/agents/{id}/memory?q=                hosszú távú memória böngészése

GET    /api/session-templates                    (CRUD ugyanígy)

POST   /api/sessions                             sablonból vagy inline konfigból
GET    /api/sessions
GET    /api/sessions/{id}
POST   /api/sessions/{id}/start | pause | resume | stop
GET    /api/sessions/{id}/events?after_seq=&types=
GET    /api/sessions/{id}/agents/{aid}/state     aktuális perszóna-állapot
GET    /api/sessions/{id}/agents/{aid}/history   persona_versions
GET    /api/sessions/{id}/agents/{aid}/memory?q=
GET    /api/sessions/{id}/approvals
POST   /api/sessions/{id}/approvals/{pid}        {"decision": "approve" | "reject"}
GET    /api/sessions/{id}/export?format=jsonl|csv
```

### 14.2 WebSocket

`/ws/sessions/{id}?participant=<id>&token=<t>&after_seq=<n>`

- Csatlakozáskor a szerver elküldi az összes `after_seq` utáni eseményt az adatbázisból, majd élőben folytatja. Így az újracsatlakozás veszteségmentes.
- Szerver → kliens: `{"type": "event", "seq": 42, "event": {...}}`, valamint `{"type": "your_turn", "deadline": "..."}` emberi résztvevőnek.
- Kliens → szerver: `{"type": "say", "content": "..."}`, `{"type": "raise_hand"}`, `{"type": "lower_hand"}`, `{"type": "intervene", "content": "..."}` (csak moderátor).

### 14.3 Hozzáférés

v1: a hub `127.0.0.1`-re köt. Emberi résztvevőnként generált token az URL-ben. Ha LAN-on is el kell érni, egy egyszerű, `.env`-ben megadott admin-token kell a konfigurációs végpontokra.

---

## 15. Web UI

| Nézet | Tartalom |
|---|---|
| Irányítópult | sessionök listája állapottal, költséggel; új session |
| Agent-könyvtár | agent-sablonok; űrlap: modell, mag, kezdő állapot, drift mód, memória, konzisztencia-ellenőrzés |
| Session-építő | téma, vitakérdések, résztvevők kiválasztása, moderátor, stratégia (sémából generált paraméterűrlap), leállási feltételek; mentés sablonként |
| Élő asztal | átirat; oldalsáv a résztvevőkkel (ki beszél, ki „gondolkodik”, kinél van jelentkezés); kör- és költségmérő; szünet/folytatás/leállítás; emberi beviteli mező és jelentkezés gomb; jóváhagyási panel APPROVED módhoz |
| Agent-inspektor | mag, aktuális állapot, álláspont-idővonal kérdésenként (Chart.js), memória-kereső, konzisztencia-események |
| Visszajátszás és export | egy lezárt session eseményenként léptethető; JSONL/CSV export |

Megvalósítás: Jinja2-sablonok, HTMX a részleges frissítésekhez, `htmx-ext-ws` az élő eseményekhez; a szerver HTML-részleteket küld az eseményekből.

---

## 16. Példa session-sablon

```yaml
title: "Lehet-e egy LLM társszerző?"
topic: >
  A generatív AI egyre nagyobb szerepet kap a tudományos cikkírásban.
  Vitassátok meg a szerzőség, a felelősség és a hozzájárulás kérdését.
language: hu
questions:
  - id: q_authorship
    text: "Feltüntethető-e egy LLM szerzőként egy tudományos cikkben?"
  - id: q_disclosure
    text: "Kötelező legyen-e részletesen közölni az AI-használatot?"
participants:
  - agent: skeptic_methodologist
  - agent: pragmatic_editor
  - agent: ai_optimist
  - human: "Sándor"
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

## 17. Tesztelés

- **Unit:** stratégiák (determinisztikus sorrend, jelentkezés elsőbbsége), drift policyk (vágás, elutasítás), RRF-fúzió, kontextus-összeállítás tokenkerettel, prompt-renderelés (snapshot-tesztek).
- **Integráció:** teljes session `FakeLLM`-mel (forgatókönyv szerinti válaszok), ideiglenes SQLite-adatbázison; ellenőrzés: eseménysorrend, projekciók, leállási feltételek.
- **Visszajátszás-teszt:** egy lefutott session eseményeiből újraépített `SessionState` egyezik a futás végi állapottal.
- **WebSocket:** `httpx` + FastAPI TestClient; újracsatlakozás `after_seq`-kel nem veszít eseményt.
- **Live:** `pytest -m live`, valódi API-val, alapból kihagyva; egy rövid, kétagentes, kétkörös vita.
- CI: ruff, pyright strict, pytest (live nélkül).

---

## 18. Mérföldkövek

| # | Tartalom | Kész, ha |
|---|---|---|
| M0 | Repo, uv, ruff, pyright, pytest; Settings; DB + Alembic; pragmák; **sqlite-vec betöltése Macen és PC-n** | a tesztek zöldek, a sqlite-vec mindkét gépen betöltődik aiosqlite alatt |
| M1 | Domain modell, események, SessionEngine, RoundRobin, AIAgent, LiteLLM-kliens, FakeLLM, CLI `roundtable run sablon.yaml` | egy 3 agentes vita lefut a parancssorban, az események az adatbázisban vannak, a visszajátszás-teszt zöld |
| M2 | FastAPI hub, WebSocket, élő asztal nézet, HumanParticipant, szünet/folytatás/leállítás, költségkorlát | böngészőből indítható vita, ember hozzászól, újratöltés után az átirat hiánytalan |
| M3 | Memória: epizodikus összefoglalók, hosszú távú tanulságok, sqlite-vec + FTS5 + RRF, EmbeddingProvider | egy második sessionben az agent hivatkozik az előző session tanulságára |
| M4 | Perszóna: mag/állapot, vitakérdések, reflexió, LOCKED + FREE + shadow, persona_versions, agent-inspektor idővonallal | FREE módban látható és indokolt álláspont-változás; LOCKED módban az állapot nem változik, a shadow javaslatok naplózódnak |
| M5 | Plugin-registry entry pointokkal, sémából generált űrlapok, HandRaisePriority, második stratégia (ModeratorPicks vagy Bidding) | egy külön telepített csomag stratégiája megjelenik a UI-ban és használható |
| M6 | BOUNDED és APPROVED mód, jóváhagyási panel, konzisztencia-ellenőrzés, export, visszajátszás nézet | minden drift mód működik; a JSONL-export elemzésre kész |
| M7 (opcionális) | RemoteAgent, token-streaming a UI-ba, eszközhasználó agentek (pl. irodalomkeresés) | |

M1 után a rendszer már kutatási kísérletekre is használható parancssorból, a UI nélkül is.

---

## 19. Kockázatok és nyitott kérdések

- **sqlite-vec betöltése:** a macOS rendszer-Pythonja és egyes Python-buildek nem engedik a `enable_load_extension`-t. Az M0 első feladata ezt ellenőrizni az uv által telepített Pythonnal, aiosqlite alatt, mindkét gépen.
- **Konvergencia és sycophancy:** az agentek hajlamosak gyorsan egyetérteni. Ellenszerek: eltérő modellek az agentekhez, erős és konkrét mag, a reflexiós prompt „nincs változás” opciója, ördögügyvéd szerep, és a BOUNDED mód.
- **Költség:** N agent × kör × (beszéd + összefoglaló + reflexió + opcionális bíró). A költségkorlát kötelező alapbeállítás; a mellékhívásokhoz (összefoglaló, reflexió, bíró) olcsóbb modell konfigurálható.
- **Többszereplős beszélgetés chat-formátumban:** a „mások = user, én = assistant” leképezés működik, de modellenként érdemes összevetni azzal, amikor az egész átirat egyetlen user-üzenetben érkezik.
- **Vitakérdések rögzítése:** v1-ben a kérdéseket a session elején kell megadni. Nyitott kérdés, hogy később a moderátor vagy az agentek vehessenek-e fel új kérdést futás közben.
- **Promptnyelv:** magyar vagy angol utasítás, magyar kimenettel; a perszóna-stabilitásra gyakorolt hatást mérni kell.
