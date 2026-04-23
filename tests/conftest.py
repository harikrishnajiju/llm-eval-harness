"""
Pytest fixtures shared across all test modules.

All external dependencies (Ollama, Ragas, HuggingFace datasets) are mocked
so the test suite runs without any running services.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

# Point DB at in-memory SQLite before any app code runs
os.environ.setdefault("DB_PATH", ":memory:")
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")


@pytest.fixture
def mock_ollama(monkeypatch):
    """
    Return a mock ChatOllama that echoes the question as the answer.
    Monkeypatches ChatOllama in all modules that import it.
    """

    class _FakeMessage:
        content = "Mock answer based on context."

    class _FakeChatOllama:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, prompt, **kwargs):
            return _FakeMessage()

    monkeypatch.setattr("rag_pipeline.chain.ChatOllama", _FakeChatOllama)
    return _FakeChatOllama()


@pytest.fixture
def mock_ragas_evaluate(monkeypatch):
    """
    Patch ragas.evaluate AND the Ragas metric constructors so evals never
    touch a real LLM.  Ragas v0.4 requires the llm at metric __init__ time,
    so we stub the classes themselves as well.
    """
    import pandas as pd

    class _FakeResult:
        def to_pandas(self):
            return pd.DataFrame(
                [
                    {
                        "context_precision": 0.8,
                        "answer_relevancy": 0.75,
                        "faithfulness": 0.9,
                    }
                ]
            )

    def _fake_evaluate(*args, **kwargs):
        return _FakeResult()

    monkeypatch.setattr("evaluator.ragas_eval.evaluate", _fake_evaluate)

    # Stub the metric classes so their __init__ doesn't require a real llm
    fake_metric = MagicMock()
    monkeypatch.setattr("evaluator.ragas_eval.ContextPrecision", lambda **kw: fake_metric)
    monkeypatch.setattr("evaluator.ragas_eval.AnswerRelevancy", lambda **kw: fake_metric)
    monkeypatch.setattr("evaluator.ragas_eval.Faithfulness", lambda **kw: fake_metric)

    return _fake_evaluate


@pytest.fixture
def mock_embeddings(monkeypatch):
    """Return a mock embeddings instance that produces random-ish vectors."""

    class _FakeEmbeddings:
        def embed_documents(self, texts):
            return [[0.1] * 384 for _ in texts]

        def embed_query(self, text):
            return [0.1] * 384

    fake = _FakeEmbeddings()
    monkeypatch.setattr("rag_pipeline.embeddings.get_embeddings", lambda: fake)
    return fake


@pytest_asyncio.fixture
async def test_client(mock_ollama, mock_ragas_evaluate, mock_embeddings):
    """
    Async HTTP client pointed at the FastAPI app with all external deps mocked.
    Uses an in-memory SQLite database.
    """
    from httpx import ASGITransport, AsyncClient

    from api.main import app
    from storage.database import init_db

    await init_db()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"X-API-Key": "test-api-key"},
    ) as client:
        yield client
