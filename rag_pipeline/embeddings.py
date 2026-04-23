"""
Embeddings provider for the RAG pipeline.

Uses sentence-transformers/all-MiniLM-L6-v2 via LangChain's HuggingFaceEmbeddings.
Model is ~80 MB and runs well on Apple Silicon M-series CPUs.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.

    The model is downloaded on first call and cached in memory for subsequent calls.
    Using lru_cache ensures we never load the model more than once per process.

    Returns:
        A LangChain HuggingFaceEmbeddings instance backed by all-MiniLM-L6-v2.
    """
    return HuggingFaceEmbeddings(
        model_name=_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
