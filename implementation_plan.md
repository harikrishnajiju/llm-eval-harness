# LLM Evaluation Harness — Implementation Plan

> **Stack:** Python · FastAPI · SQLite · Ragas · Ollama · LangChain  
> **Target:** MacBook Air M4 (local dev) · Railway free tier (API deployment)  

---

## Architecture Decisions

Before touching code, here are the key decisions baked into this plan and why:

**FAISS over Chroma or Pinecone.** FAISS runs fully in-memory with no server process. On Railway free tier, every MB matters. Chroma requires a persistent server; Pinecone costs money. FAISS is the right call for an eval harness where the vector store is rebuilt per run anyway.

**`rajpurkar/squad` as the HuggingFace dataset.** SQuAD has clean (question, context, ground_truth) triples out of the box — exactly the shape Ragas needs. No preprocessing gymnastics. Slice to 25–50 samples per run so evals complete in under 5 minutes locally.

**Ollama stays local, Railway gets only the FastAPI service.** Ollama cannot run on Railway free tier (no GPU, RAM limits). The FastAPI service stores run history, exposes endpoints, and is Railway-deployed. `OLLAMA_BASE_URL` is an env var so you can point it at your M4 locally or swap in a HuggingFace inference endpoint for CI.

**Ollama as the Ragas judge LLM too.** Ragas needs an LLM to score faithfulness and relevancy. Wire it to the same local Llama 3 instance. In GitHub Actions CI, replace it with `HuggingFaceHub` (free inference API) so CI never needs your laptop running.

**BackgroundTasks over Celery/Redis.** A full task queue is overkill here. FastAPI's built-in `BackgroundTasks` runs the eval async, returns a `run_id` immediately, and stores results to SQLite when done. Zero extra infrastructure.

**SQLAlchemy async with aiosqlite.** FastAPI is async-native. Blocking SQLite calls inside async endpoints will freeze the server. `aiosqlite` + SQLAlchemy's async engine solves this cleanly.

**Prompt variant as a first-class field.** `prompt_variant` is stored on every run record. This lets you POST two runs with different system prompts and compare Ragas scores in the `/runs` list — the core value of an eval harness.

---

## Project Structure

```
llm-eval-harness/
├── data_loader/
│   ├── __init__.py
│   └── hf_dataset.py           # Load & slice HuggingFace SQuAD dataset
├── rag_pipeline/
│   ├── __init__.py
│   ├── embeddings.py            # HuggingFace embeddings via LangChain
│   ├── vectorstore.py           # FAISS in-memory vector store builder
│   └── chain.py                 # LangChain RetrievalQA with Ollama Llama 3
├── evaluator/
│   ├── __init__.py
│   └── ragas_eval.py            # Ragas metrics runner (precision, relevancy, faithfulness)
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entry point, lifespan, middleware
│   ├── routes.py                # /runs POST+GET, /runs/{id} GET, /health GET
│   └── schemas.py               # Pydantic request/response models
├── storage/
│   ├── __init__.py
│   ├── database.py              # Async SQLite engine, session factory, init_db()
│   └── models.py                # SQLAlchemy ORM EvalRun model
├── scripts/
│   └── smoke_eval.py            # Minimal 5-sample eval for CI smoke test
├── tests/
│   ├── conftest.py              # Pytest fixtures, mocked Ollama + Ragas
│   ├── test_data_loader.py
│   ├── test_rag_pipeline.py
│   ├── test_evaluator.py
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── eval.yml             # GitHub Actions: test + smoke eval on push, deploy on main
├── Dockerfile
├── railway.toml
├── .env.example
├── requirements.txt
└── README.md
```

---

## Phase 0 — Scaffold & README

**Goal:** Every file and directory exists. The project is importable. README explains how to run locally and what each module does.

### Tasks

1. Create the full directory tree above with empty `__init__.py` files.
2. Write `README.md` covering: local setup, how to start Ollama, how to trigger an eval run via curl, how to read results.
3. Write `.env.example`:

```env
OLLAMA_BASE_URL=http://localhost:11434
HF_DATASET_NAME=rajpurkar/squad
HF_DATASET_SPLIT=validation
N_SAMPLES=25
DB_PATH=./data/evals.db
API_KEY=your-secret-key-here
```

4. Write `requirements.txt` with pinned versions:

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-huggingface>=0.0.3
langchain-ollama>=0.1.0
ragas>=0.1.14
datasets>=2.19.0
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
sqlalchemy>=2.0.0
aiosqlite>=0.20.0
pydantic-settings>=2.2.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
```

> **M4 note:** `faiss-cpu` installs cleanly on Apple Silicon. Do not use `faiss-gpu`.

---

## Phase 1 — Data Loader

**File:** `data_loader/hf_dataset.py`

**Goal:** Load SQuAD validation split, slice to `n_samples`, return a list of typed dicts ready for the RAG pipeline.

### What to implement

```python
# Signature to implement
def load_qa_dataset(
    dataset_name: str = "rajpurkar/squad",
    split: str = "validation",
    n_samples: int = 25,
    seed: int = 42,
) -> list[dict]:
    """
    Returns a list of dicts, each with keys:
      - question: str
      - context: str
      - ground_truth: str  (first answer from SQuAD answers field)
    """
```

### Notes

- Use `datasets.load_dataset()` with `trust_remote_code=False`.
- Shuffle with `seed` before slicing so results are reproducible.
- SQuAD's `answers` field is `{"text": [...], "answer_start": [...]}` — take `answers["text"][0]` as `ground_truth`.
- Cache the dataset locally by default (HF datasets handles this automatically).

---

## Phase 2 — RAG Pipeline

### `rag_pipeline/embeddings.py`

Use `sentence-transformers/all-MiniLM-L6-v2` via `langchain_huggingface.HuggingFaceEmbeddings`. This model is small (80MB), runs fast on M4 CPU, and produces good retrieval quality for English QA.

```python
# Signature to implement
def get_embeddings() -> HuggingFaceEmbeddings:
    """Returns a cached embeddings instance."""
```

### `rag_pipeline/vectorstore.py`

Build a FAISS vector store from a list of context strings. This is rebuilt fresh per eval run — no persistence needed.

```python
# Signature to implement
def build_vectorstore(contexts: list[str], embeddings) -> FAISS:
    """
    Embeds all context strings and returns a FAISS retriever.
    Use FAISS.from_texts(contexts, embeddings).
    """
```

### `rag_pipeline/chain.py`

This is the core of the pipeline. Build a `RetrievalQA` chain using `ChatOllama` and the FAISS retriever. The `prompt_variant` parameter controls the system prompt template — this is what you A/B test across runs.

```python
# Signatures to implement
def build_rag_chain(vectorstore: FAISS, prompt_variant: str = "default") -> RetrievalQA:
    """
    prompt_variant options:
      "default"  — standard "Answer using only the context" prompt
      "concise"  — instructs model to answer in one sentence
      "cot"      — chain-of-thought: "think step by step before answering"
    
    Returns a LangChain RetrievalQA chain.
    """

def run_pipeline(
    chain: RetrievalQA,
    dataset_rows: list[dict],
) -> list[dict]:
    """
    Runs each question through the chain.
    Returns list of dicts with keys:
      - question: str
      - answer: str           (model output)
      - contexts: list[str]   (retrieved chunks — needed by Ragas)
      - ground_truth: str
    """
```

> **Important:** Ragas requires `contexts` to be a list of strings (the retrieved document contents), not LangChain Document objects. Extract `.page_content` from each retrieved doc before returning.

---

## Phase 3 — Ragas Evaluator

**File:** `evaluator/ragas_eval.py`

**Goal:** Take the pipeline outputs and return a dict of scalar metric scores.

### Ragas metrics to implement

| Metric | What it measures |
|---|---|
| `context_precision` | Are the retrieved contexts actually relevant to the question? |
| `answer_relevancy` | Does the answer address the question (regardless of correctness)? |
| `faithfulness` | Is the answer grounded in the retrieved contexts (no hallucination)? |

### What to implement

```python
# Signature to implement
def run_ragas_eval(
    pipeline_outputs: list[dict],
    llm_judge,          # Pass the same ChatOllama instance used in the chain
    embeddings,         # Pass the same HuggingFaceEmbeddings instance
) -> dict[str, float]:
    """
    Converts pipeline_outputs into a Ragas EvaluationDataset,
    runs the three metrics, returns:
    {
        "context_precision": 0.82,
        "answer_relevancy": 0.76,
        "faithfulness": 0.91,
    }
    """
```

### Implementation notes

- Use `ragas.evaluate()` with `metrics=[ContextPrecision(), AnswerRelevancy(), Faithfulness()]`.
- Pass `llm=llm_judge` and `embeddings=embeddings` to `evaluate()` to avoid Ragas defaulting to OpenAI.
- Ragas returns a `Result` object — call `.to_pandas()` then take column means for scalar scores.
- Wrap in try/except: Ragas can fail on individual rows if the LLM returns malformed JSON. Log and skip bad rows rather than crashing the whole run.

### CI judge LLM

In GitHub Actions, Ollama is not available. Use this swap:

```python
from langchain_huggingface import HuggingFaceEndpoint

ci_judge = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
)
```

Control via `USE_HF_JUDGE=true` env var in CI.

---

## Phase 4 — Storage

### `storage/models.py`

```python
# SQLAlchemy ORM model to implement
class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: str                  # UUID, primary key
    model_name: str          # e.g. "llama3"
    prompt_variant: str      # e.g. "default", "concise", "cot"
    n_samples: int
    status: str              # "running" | "complete" | "failed"
    metrics: str             # JSON string of {metric: score} dict
    error: str | None        # Error message if status == "failed"
    duration_seconds: float | None
    created_at: datetime
    completed_at: datetime | None
```

### `storage/database.py`

```python
# Functions to implement
async def init_db() -> None:
    """Create tables if they don't exist. Called on FastAPI startup."""

async def save_run(run: EvalRun) -> None:
    """Insert or update a run record."""

async def get_run(run_id: str) -> EvalRun | None:
    """Fetch a single run by ID."""

async def list_runs(limit: int = 50) -> list[EvalRun]:
    """Return most recent runs, ordered by created_at DESC."""
```

Use `sqlalchemy.ext.asyncio.create_async_engine` with `"sqlite+aiosqlite:///path/to/evals.db"`.

---

## Phase 5 — FastAPI

### `api/schemas.py`

```python
# Pydantic models to implement

class EvalRequest(BaseModel):
    model_name: str = "llama3"
    prompt_variant: Literal["default", "concise", "cot"] = "default"
    n_samples: int = Field(default=25, ge=5, le=200)

class RunStatus(BaseModel):
    run_id: str
    status: str
    created_at: datetime

class RunDetail(BaseModel):
    run_id: str
    model_name: str
    prompt_variant: str
    n_samples: int
    status: str
    metrics: dict[str, float] | None
    error: str | None
    duration_seconds: float | None
    created_at: datetime
    completed_at: datetime | None
```

### `api/routes.py` — endpoint specification

```
POST  /runs
  Auth: X-API-Key header required
  Body: EvalRequest
  Action: Creates run record with status="running", fires BackgroundTask
  Returns: RunStatus (run_id + "running")

GET   /runs
  Auth: X-API-Key header required
  Returns: list[RunDetail], ordered by created_at DESC, limit 50

GET   /runs/{run_id}
  Auth: X-API-Key header required
  Returns: RunDetail (full record including metrics once complete)

GET   /health
  No auth
  Returns: { "status": "ok", "ollama_reachable": bool }
  Note: /health pings OLLAMA_BASE_URL/api/tags to check connectivity
```

### `api/main.py`

```python
# Structure to implement
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()           # Create tables on startup
    yield

app = FastAPI(title="LLM Eval Harness", lifespan=lifespan)
app.include_router(router)
```

### Auth dependency

```python
# Simple API key check — add as a FastAPI dependency
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
```

---

## Phase 6 — GitHub Actions

**File:** `.github/workflows/eval.yml`

```yaml
name: Eval CI

on:
  push:
    branches: ["**"]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - run: pip install -r requirements.txt

      - name: Run unit tests (mocked LLM)
        run: pytest tests/ -v --tb=short
        env:
          USE_HF_JUDGE: "true"
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          DB_PATH: ":memory:"
          API_KEY: "ci-test-key"

      - name: Smoke eval (5 samples, HF judge)
        run: python scripts/smoke_eval.py
        env:
          USE_HF_JUDGE: "true"
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          N_SAMPLES: "5"

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: railway/deploy-action@v1
        with:
          service: llm-eval-harness
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
```

### `scripts/smoke_eval.py`

A standalone script that runs a minimal 5-sample eval end-to-end (data load → RAG → Ragas → print scores). No FastAPI, no SQLite. Used in CI to verify the core pipeline works.

---

## Phase 7 — Dockerfile & Railway Config

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# SQLite data directory
RUN mkdir -p /data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `railway.toml`

```toml
[build]
builder = "dockerfile"

[deploy]
startCommand = "uvicorn api.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 30
restartPolicyType = "on_failure"

[[volumes]]
mountPath = "/data"
```

### Railway environment variables to set in dashboard

```
OLLAMA_BASE_URL     = http://your-local-ip:11434  # or leave unset if eval runs locally only
DB_PATH             = /data/evals.db
API_KEY             = <generate a strong secret>
HF_DATASET_NAME     = rajpurkar/squad
N_SAMPLES           = 25
```

> **Railway free tier limit:** 512MB RAM. The `sentence-transformers` model uses ~200MB. SQuAD dataset cache uses ~50MB. You're within limits for the API service. Do not run eval workloads on Railway — only serve the API endpoints there.

---

## Testing Strategy

### Unit tests (run in CI without any real LLM)

| Test file | What to mock | What to assert |
|---|---|---|
| `test_data_loader.py` | `datasets.load_dataset` | Returns correct keys, correct slice size |
| `test_rag_pipeline.py` | `ChatOllama`, `FAISS` | Chain returns `answer` + `contexts` keys |
| `test_evaluator.py` | `ragas.evaluate` | Returns dict with all three metric keys, values 0–1 |
| `test_api.py` | Full pipeline via `AsyncClient` | POST /runs returns run_id; GET /runs lists it |

### `tests/conftest.py` fixtures to write

```python
@pytest.fixture
def mock_ollama():
    """Returns a mock ChatOllama that echoes the question as the answer."""

@pytest.fixture  
def mock_ragas_evaluate():
    """Returns fixed scores: {context_precision: 0.8, answer_relevancy: 0.75, faithfulness: 0.9}"""

@pytest.fixture
async def test_client(mock_ollama, mock_ragas_evaluate):
    """AsyncClient with in-memory SQLite DB."""
```

---

## Local Development Workflow

```bash
# 1. Start Ollama and pull the model (one time)
brew install ollama
ollama serve &
ollama pull llama3

# 2. Clone, create venv, install deps
git clone <repo>
cd llm-eval-harness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Copy and fill env
cp .env.example .env

# 4. Start the API server
uvicorn api.main:app --reload --port 8000

# 5. Trigger an eval run
curl -X POST http://localhost:8000/runs \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "llama3", "prompt_variant": "default", "n_samples": 10}'

# 6. Poll for results
curl http://localhost:8000/runs/<run_id> \
  -H "X-API-Key: your-secret-key-here"

# 7. Compare prompt variants
curl -X POST http://localhost:8000/runs \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "llama3", "prompt_variant": "cot", "n_samples": 10}'

curl http://localhost:8000/runs \
  -H "X-API-Key: your-secret-key-here"
```

---

## README Content

The `README.md` should cover:

1. **What this is** — one paragraph: LLM eval harness for RAG pipelines using Ragas metrics, designed to A/B test prompt variants against Llama 3 locally.
2. **Prerequisites** — Python 3.11+, Ollama installed, Git.
3. **Local setup** — the 6-step workflow above.
4. **API reference** — the four endpoints with curl examples.
5. **Understanding the metrics** — one sentence each on context_precision, answer_relevancy, faithfulness.
6. **Prompt variants** — what `default`, `concise`, and `cot` do differently and when to use each.
7. **Deployment** — how to deploy to Railway, which env vars to set.
8. **CI** — what the GitHub Actions workflow does, what secrets to add.

---

## Build Order

Follow this sequence. Each phase is a clean dependency on the previous one:

```
Phase 0 (scaffold)
    → Phase 1 (data loader)         test: load 5 rows, print them
    → Phase 2 (RAG pipeline)        test: ask one question, print answer
    → Phase 3 (evaluator)           test: score one question, print metrics
    → Phase 4 (storage)             test: save one run, read it back
    → Phase 5 (FastAPI)             test: curl POST /runs, GET /runs
    → Phase 6 (GitHub Actions)      test: push to branch, watch CI pass
    → Phase 7 (Railway)             test: deploy, curl the live URL
```

Validate each phase with a quick manual test before moving to the next. The evaluator in Phase 3 is the riskiest — Ragas + local LLM judge is the most likely source of unexpected errors. Give it extra time.