"""
FAISS in-memory vector store builder.

Contexts are embedded and indexed fresh per eval run — no persistence needed.
This keeps the architecture simple and avoids stale index issues.
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS  # type: ignore


def build_vectorstore(contexts: list[str], embeddings) -> FAISS:
    """
    Embed all context strings and return a FAISS vector store.

    The index is built in memory and is not persisted to disk.
    It should be used immediately to build a retriever for the RAG chain.

    Args:
        contexts:   List of text passages to index.
        embeddings: A LangChain Embeddings instance (e.g. HuggingFaceEmbeddings).

    Returns:
        A FAISS vector store ready for similarity search.
    """
    if not contexts:
        raise ValueError("contexts list must not be empty")

    return FAISS.from_texts(contexts, embeddings)
