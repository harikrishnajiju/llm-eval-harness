# 🚀 LLM Eval Harness

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Ragas](https://img.shields.io/badge/Ragas-Modern_Eval-FF6F61?style=flat)](https://docs.ragas.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Llama3-white?style=flat&logo=ollama)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A high-performance, self-hosted evaluation harness for RAG pipelines.** 

Run **Ragas** metrics against **Llama 3** locally, A/B test prompt variants, and store results in a lightweight SQLite database — all without sending your sensitive data to external APIs.

---

## 🌟 Features

- **Local Inference:** Fully integrated with **Ollama** for private, local LLM execution.
- **Modern Evaluation:** Uses **Ragas 0.4.x** modern component architecture for high-fidelity scoring.
- **A/B Testing:** Easily compare `default`, `concise`, and `Chain-of-Thought (CoT)` prompt variants.
- **Asynchronous Pipeline:** Background task execution for evaluation runs with persistent storage.
- **Production-Ready:** Built with **FastAPI**, **SQLAlchemy (Async)**, and **Pydantic**.
- **CI/CD Integrated:** Ready for **GitHub Actions** and **Railway** deployment.

---

## 🛠️ Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com/)** installed and running.
- **Llama 3** pulled: `ollama pull llama3`

---

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/llm-eval-harness.git
cd llm-eval-harness

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` to set your secrets:
```env
OLLAMA_BASE_URL=http://localhost:11434
API_KEY=your-secure-secret-key
HF_DATASET_NAME=rajpurkar/squad
```

### 3. Run the Server

> **Important:** To support Ragas's nested event loop, you **must** start the server using the standard `asyncio` loop:

```bash
uvicorn api.main:app --reload --port 8000 --loop asyncio
```

---

## 📊 API Usage

### 🚀 Trigger an Evaluation Run

```bash
curl -X POST http://localhost:8000/runs \
  -H "X-API-Key: your-secure-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "llama3",
    "prompt_variant": "cot",
    "n_samples": 5
  }'
```

### 📈 Monitor Results

Visit the interactive documentation: **[http://localhost:8000/docs](http://localhost:8000/docs)** (Authorize with your `API_KEY`).

Or poll via `curl`:
```bash
curl http://localhost:8000/runs/<run_id> \
  -H "X-API-Key: your-secure-secret-key"
```

---

## 🔍 Understanding the Metrics

| Metric | What it measures | Why it matters |
|---|---|---|
| **Context Precision** | Retriever Relevance | High = Your retriever is pulling the right chunks. |
| **Answer Relevancy** | Semantic Alignment | High = The model's answer actually addresses the user's question. |
| **Faithfulness** | Factuality/Grounding | High = The model is not hallucinating and stays true to the context. |

---

## 📁 Project Structure

```text
llm-eval-harness/
├── api/                # FastAPI app & REST routes
├── evaluator/          # Ragas evaluation logic & custom metrics
├── rag_pipeline/       # LangChain + FAISS + Ollama chain
├── storage/            # SQLAlchemy async SQLite persistence
├── data_loader/        # SQuAD dataset integration
├── scripts/            # CI smoke tests & utilities
├── tests/              # Unit & Integration tests
└── .github/workflows/  # CI/CD pipeline
```

---

## 🚢 Deployment (Railway)

1. Push this repo to GitHub.
2. Link the repository to a new **Railway** project.
3. Set your environment variables (`API_KEY`, `OLLAMA_BASE_URL`, etc.).
4. Railway will automatically build and deploy using the included `Dockerfile`.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
