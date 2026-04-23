"""
Smoke eval script — runs a minimal 5-sample eval end-to-end and prints scores.

Used in CI (GitHub Actions) to verify the full pipeline works.
No FastAPI, no SQLite — just the core pipeline components.

Usage:
    USE_HF_JUDGE=true HF_TOKEN=... N_SAMPLES=5 python scripts/smoke_eval.py
"""

from __future__ import annotations

import os
import sys

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    n_samples = int(os.getenv("N_SAMPLES", "5"))
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("MODEL_NAME", "llama3")
    use_hf_judge = os.getenv("USE_HF_JUDGE", "false").lower() == "true"

    print(f"[smoke_eval] Starting eval: model={model_name}, n_samples={n_samples}")
    print(f"[smoke_eval] use_hf_judge={use_hf_judge}")

    # --- Phase 1: Load dataset ---
    print("[smoke_eval] Loading dataset…")
    from data_loader.hf_dataset import load_qa_dataset

    rows = load_qa_dataset(n_samples=n_samples)
    print(f"[smoke_eval] Loaded {len(rows)} rows")

    # --- Phase 2: Build RAG pipeline ---
    print("[smoke_eval] Building embeddings and vector store…")
    from rag_pipeline.embeddings import get_embeddings
    from rag_pipeline.vectorstore import build_vectorstore
    from rag_pipeline.chain import build_rag_chain, run_pipeline

    embeddings = get_embeddings()
    contexts = [row["context"] for row in rows]
    vectorstore = build_vectorstore(contexts, embeddings)

    chain = build_rag_chain(
        vectorstore=vectorstore,
        prompt_variant="default",
        model_name=model_name,
        ollama_base_url=ollama_url,
    )

    print("[smoke_eval] Running pipeline…")
    pipeline_outputs = run_pipeline(chain, rows)
    print(f"[smoke_eval] Got {len(pipeline_outputs)} pipeline outputs")

    if not pipeline_outputs:
        print("[smoke_eval] ERROR: No pipeline outputs — check Ollama connectivity")
        sys.exit(1)

    # --- Phase 3: Ragas evaluation ---
    print("[smoke_eval] Running Ragas evaluation…")
    from evaluator.ragas_eval import run_ragas_eval

    if use_hf_judge:
        from langchain_huggingface import HuggingFaceEndpoint

        hf_token = os.environ["HF_TOKEN"]
        llm_judge = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            huggingfacehub_api_token=hf_token,
        )
    else:
        from langchain_ollama import ChatOllama

        llm_judge = ChatOllama(model=model_name, base_url=ollama_url)

    scores = run_ragas_eval(pipeline_outputs, llm_judge=llm_judge, embeddings=embeddings)

    print("\n[smoke_eval] ✅ Eval complete. Scores:")
    for metric, score in scores.items():
        print(f"  {metric:25s}: {score:.4f}" if score == score else f"  {metric:25s}: NaN")

    # Fail CI if all scores are NaN
    import math

    if all(math.isnan(v) for v in scores.values()):
        print("[smoke_eval] ERROR: All scores are NaN — evaluation failed")
        sys.exit(1)

    print("[smoke_eval] Done.")


if __name__ == "__main__":
    main()
