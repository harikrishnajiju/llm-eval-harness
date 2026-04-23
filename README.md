# LLM Eval Harness

A self-hosted evaluation harness for RAG pipelines. Run Ragas metrics against Llama 3 locally, A/B test prompt variants, and store results in a lightweight SQLite database — all without sending data to external APIs.

---

## What this is

This project wraps a [LangChain](https://python.langchain.com/) RetrievalQA pipeline (backed by [Ollama](https://ollama.com/) + FAISS) with a [Ragas](https://docs.ragas.io/) evaluation loop and a [FastAPI](https://fastapi.tiangolo.com/) REST API. You POST an eval request, it runs in the background, and you poll for results. The core value is **comparing prompt variants**: send the same 25 SQuAD questions through a `default`, `concise`, and `cot` prompt and see which one gets better faithfulness or context precision scores.

---

## Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com/)** installed and the model pulled (see Step 1 below)
- **Git**

---

## Local Setup

```bash
# 1. Install Ollama, start it, pull the model (one time)
brew install ollama
ollama serve &
ollama pull llama3

# 2. Clone and set up the project
git clone <repo-url>
cd llm-eval-harness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and set API_KEY to a secret of your choice

# 4. Start the API server
uvicorn api.main:app --reload --port 8000

# 5. Trigger an eval run
curl -X POST http://localhost:8000/runs \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "llama3", "prompt_variant": "default", "n_samples": 10}'

# 6. Poll for results (replace <run_id> with the ID from step 5)
curl http://localhost:8000/runs/<run_id> \
  -H "X-API-Key: your-secret-key-here"

# 7. Compare prompt variants
curl -X POST http://localhost:8000/runs \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "llama3", "prompt_variant": "cot", "n_samples": 10}'

# 8. List all runs
curl http://localhost:8000/runs \
  -H "X-API-Key: your-secret-key-here"
```

---

## API Reference

All endpoints except `/health` require the `X-API-Key` header.

### `POST /runs`
Trigger a new eval run (returns immediately; eval runs in background).

**Request body:**
```json
{
  "model_name": "llama3",
  "prompt_variant": "default",
  "n_samples": 25
}
```

**Response `202`:**
```json
{
  "run_id": "uuid",
  "status": "running",
  "created_at": "2024-01-01T00:00:00Z"
}
```

---

### `GET /runs`
List the 50 most recent eval runs, ordered newest first.

**Response `200`:** Array of run detail objects (see below).

---

### `GET /runs/{run_id}`
Get full details for a specific run, including metrics once complete.

**Response `200`:**
```json
{
  "run_id": "uuid",
  "model_name": "llama3",
  "prompt_variant": "default",
  "n_samples": 25,
  "status": "complete",
  "metrics": {
    "context_precision": 0.82,
    "answer_relevancy": 0.76,
    "faithfulness": 0.91
  },
  "error": null,
  "duration_seconds": 187.4,
  "created_at": "2024-01-01T00:00:00Z",
  "completed_at": "2024-01-01T00:03:07Z"
}
```

`status` is one of: `"running"` | `"complete"` | `"failed"`

---

### `GET /health`
No auth required. Returns `{ "status": "ok", "ollama_reachable": true/false }`.

---

## Understanding the Metrics

| Metric | What it measures |
|---|---|
| **context_precision** | Are the retrieved passages actually relevant to the question? High = retriever is pulling the right chunks. |
| **answer_relevancy** | Does the answer address the question, regardless of factual correctness? High = model is on-topic. |
| **faithfulness** | Is the answer grounded in the retrieved context with no hallucination? High = model is not making things up. |

All scores are floats between 0 (worst) and 1 (best).

---

## Prompt Variants

| Variant | Instruction style | When to use |
|---|---|---|
| `default` | "Answer using only the context." | Baseline; good general-purpose. |
| `concise` | "Answer in one sentence." | When you want terse, high-precision answers. |
| `cot` | "Think step by step before answering." | When faithfulness matters most; the model reasons explicitly. |

Compare variants by running two `POST /runs` calls with the same `n_samples` and `model_name` but different `prompt_variant` values, then comparing the metrics in the results.

---

## Deployment (Railway)

1. Push this repo to GitHub.
2. Create a new Railway project and link to this repo.
3. Railway detects the `Dockerfile` automatically.
4. Set these environment variables in the Railway dashboard:

```
OLLAMA_BASE_URL     = http://<your-local-ip>:11434  # or leave unset
DB_PATH             = /data/evals.db
API_KEY             = <generate a strong secret>
HF_DATASET_NAME     = rajpurkar/squad
N_SAMPLES           = 25
```

5. Railway will deploy on every push to `main`.

> **Note:** The Railway free tier has 512 MB RAM. This is enough to serve the API endpoints. Do **not** run eval workloads on Railway — trigger them from your local machine (or a machine with Ollama running) and let the API store and serve the results.

---

## CI (GitHub Actions)

The workflow in `.github/workflows/eval.yml`:
- **On every push / PR:** Runs the full unit test suite (all external deps mocked, no Ollama needed) + a 5-sample smoke eval using the HuggingFace Inference API as the judge LLM.
- **On push to `main`:** Deploys to Railway after tests pass.

**Secrets to add in GitHub Settings → Secrets:**
- `HF_TOKEN` — HuggingFace API token (free, used as CI judge LLM)
- `RAILWAY_TOKEN` — Railway deploy token (from Railway dashboard → Account → Tokens)

---

## Project Structure

```
llm-eval-harness/
├── data_loader/        # HuggingFace SQuAD dataset loader
├── rag_pipeline/       # Embeddings, FAISS vector store, LangChain chain
├── evaluator/          # Ragas metrics runner
├── api/                # FastAPI app, routes, schemas
├── storage/            # SQLAlchemy async SQLite engine + CRUD
├── scripts/            # smoke_eval.py — CI smoke test
├── tests/              # Unit + integration tests (fully mocked)
├── .github/workflows/  # GitHub Actions CI/CD
├── Dockerfile
├── railway.toml
├── requirements.txt
└── .env.example
```
