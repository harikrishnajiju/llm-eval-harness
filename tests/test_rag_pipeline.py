"""
Tests for the RAG pipeline (embeddings, vectorstore, chain).

All external calls (Ollama, FAISS, sentence-transformers) are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Vectorstore tests
# ---------------------------------------------------------------------------


def test_build_vectorstore_raises_on_empty():
    """build_vectorstore should raise ValueError on empty context list."""
    from rag_pipeline.vectorstore import build_vectorstore

    fake_embeddings = MagicMock()
    with pytest.raises(ValueError, match="contexts list must not be empty"):
        build_vectorstore([], fake_embeddings)


@patch("rag_pipeline.vectorstore.FAISS")
def test_build_vectorstore_calls_from_texts(mock_faiss):
    """build_vectorstore must delegate to FAISS.from_texts."""
    from rag_pipeline.vectorstore import build_vectorstore

    fake_embeddings = MagicMock()
    contexts = ["Context A", "Context B"]

    build_vectorstore(contexts, fake_embeddings)

    mock_faiss.from_texts.assert_called_once_with(contexts, fake_embeddings)


# ---------------------------------------------------------------------------
# Chain tests
# ---------------------------------------------------------------------------


def test_build_rag_chain_invalid_variant():
    """build_rag_chain should raise ValueError for unknown prompt_variant."""
    from unittest.mock import MagicMock

    from rag_pipeline.chain import build_rag_chain

    fake_vs = MagicMock()
    with pytest.raises(ValueError, match="Unknown prompt_variant"):
        build_rag_chain(fake_vs, prompt_variant="bogus")


@patch("rag_pipeline.chain.ChatOllama")
@patch("rag_pipeline.chain.RetrievalQA")
def test_build_rag_chain_returns_chain(mock_rqa, mock_ollama):
    """build_rag_chain should return the result of RetrievalQA.from_chain_type."""
    from rag_pipeline.chain import build_rag_chain

    fake_vs = MagicMock()
    fake_vs.as_retriever.return_value = MagicMock()

    chain = build_rag_chain(fake_vs, prompt_variant="default")

    assert chain == mock_rqa.from_chain_type.return_value


@patch("rag_pipeline.chain.ChatOllama")
@patch("rag_pipeline.chain.RetrievalQA")
def test_run_pipeline_returns_expected_keys(mock_rqa, mock_ollama):
    """run_pipeline outputs must contain question, answer, contexts, ground_truth."""
    from langchain_core.documents import Document  # type: ignore

    from rag_pipeline.chain import run_pipeline

    # Wire up the mock chain
    fake_doc = MagicMock(spec=Document)
    fake_doc.page_content = "Retrieved context text."

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "result": "Mock answer.",
        "source_documents": [fake_doc],
    }

    dataset_rows = [
        {"question": "What is AI?", "context": "AI context.", "ground_truth": "AI answer."}
    ]

    results = run_pipeline(mock_chain, dataset_rows)

    assert len(results) == 1
    row = results[0]
    assert "question" in row
    assert "answer" in row
    assert "contexts" in row
    assert "ground_truth" in row
    assert isinstance(row["contexts"], list)
    assert row["contexts"][0] == "Retrieved context text."


@patch("rag_pipeline.chain.ChatOllama")
@patch("rag_pipeline.chain.RetrievalQA")
def test_run_pipeline_skips_failed_rows(mock_rqa, mock_ollama):
    """run_pipeline must skip rows that raise, not crash the whole run."""
    from rag_pipeline.chain import run_pipeline

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = RuntimeError("Ollama timed out")

    dataset_rows = [
        {"question": "Q?", "context": "C.", "ground_truth": "A."},
        {"question": "Q2?", "context": "C2.", "ground_truth": "A2."},
    ]

    results = run_pipeline(mock_chain, dataset_rows)

    # All rows failed — result list is empty, not an exception
    assert results == []
