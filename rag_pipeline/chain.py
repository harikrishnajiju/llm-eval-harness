"""
RAG chain builder and pipeline runner.

Supports three prompt variants that can be A/B tested across eval runs:
  - "default" : standard "answer using only the context" instruction
  - "concise"  : instructs the model to answer in one sentence
  - "cot"      : chain-of-thought — think step by step before answering
"""

from __future__ import annotations

import logging

from langchain_classic.chains import RetrievalQA  # type: ignore
from langchain_classic.prompts import PromptTemplate  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_ollama import ChatOllama  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates per variant
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATES: dict[str, str] = {
    "default": (
        "Use ONLY the following context to answer the question. "
        "If the answer is not in the context, say 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "concise": (
        "Use ONLY the following context to answer the question in ONE sentence. "
        "If the answer is not in the context, say 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (one sentence):"
    ),
    "cot": (
        "Use ONLY the following context to answer the question. "
        "Think step by step before giving your final answer. "
        "If the answer is not in the context, say 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Step-by-step reasoning and answer:"
    ),
}

_VALID_VARIANTS = frozenset(_PROMPT_TEMPLATES.keys())


def build_rag_chain(
    vectorstore: FAISS,
    prompt_variant: str = "default",
    model_name: str = "llama3",
    ollama_base_url: str = "http://localhost:11434",
) -> RetrievalQA:
    """
    Build a LangChain RetrievalQA chain backed by Ollama and FAISS.

    Args:
        vectorstore:      A FAISS vector store containing indexed contexts.
        prompt_variant:   One of "default", "concise", or "cot".
        model_name:       Ollama model tag (e.g. "llama3").
        ollama_base_url:  Base URL of the running Ollama server.

    Returns:
        A configured RetrievalQA chain.

    Raises:
        ValueError: If prompt_variant is not recognized.
    """
    if prompt_variant not in _VALID_VARIANTS:
        raise ValueError(
            f"Unknown prompt_variant '{prompt_variant}'. "
            f"Must be one of: {sorted(_VALID_VARIANTS)}"
        )

    llm = ChatOllama(model=model_name, base_url=ollama_base_url)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate(
        template=_PROMPT_TEMPLATES[prompt_variant],
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


def run_pipeline(
    chain: RetrievalQA,
    dataset_rows: list[dict],
) -> list[dict]:
    """
    Run each dataset row through the RAG chain and collect outputs.

    Args:
        chain:        A RetrievalQA chain from build_rag_chain().
        dataset_rows: List of dicts with keys: question, context, ground_truth.

    Returns:
        List of dicts with keys:
          - question:     str
          - answer:       str   (model output)
          - contexts:     list[str]  (retrieved chunk contents — needed by Ragas)
          - ground_truth: str
    """
    results: list[dict] = []

    for i, row in enumerate(dataset_rows):
        logger.info("Running pipeline on row %d/%d", i + 1, len(dataset_rows))
        try:
            response = chain.invoke({"query": row["question"]})

            # Extract string content from retrieved Document objects
            source_docs = response.get("source_documents", [])
            contexts = [doc.page_content for doc in source_docs]

            results.append(
                {
                    "question": row["question"],
                    "answer": response["result"],
                    "contexts": contexts,
                    "ground_truth": row["ground_truth"],
                }
            )
        except Exception as exc:
            logger.warning("Pipeline failed on row %d: %s", i, exc, exc_info=True)
            # Skip failed rows rather than crashing the whole run
            continue

    return results
