"""
Ragas evaluator — scores RAG pipeline outputs using three metrics:

  context_precision  — are the retrieved chunks relevant to the question?
  answer_relevancy   — does the answer actually address the question?
  faithfulness       — is the answer grounded in the retrieved context?
"""

from __future__ import annotations

import logging
import os

from datasets import Dataset  # type: ignore
from ragas import evaluate  # type: ignore
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness  # type: ignore

logger = logging.getLogger(__name__)


def _get_judge_llm(llm_judge=None):
    """
    Return the LLM to use as the Ragas judge.

    If USE_HF_JUDGE=true is set (CI environment), swap to a HuggingFace
    endpoint so CI never requires a running Ollama server.
    If llm_judge is provided, use it directly.
    """
    if llm_judge is not None:
        return llm_judge

    if os.getenv("USE_HF_JUDGE", "").lower() == "true":
        from langchain_huggingface import HuggingFaceEndpoint  # type: ignore

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError(
                "USE_HF_JUDGE=true but HF_TOKEN env var is not set."
            )
        return HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            huggingfacehub_api_token=hf_token,
        )

    return None  # Ragas will use its default (requires OPENAI_API_KEY — only for tests)


def run_ragas_eval(
    pipeline_outputs: list[dict],
    llm_judge=None,
    embeddings=None,
) -> dict[str, float]:
    """
    Score pipeline outputs using the three Ragas metrics.

    Args:
        pipeline_outputs: List of dicts with keys:
                            question, answer, contexts (list[str]), ground_truth.
        llm_judge:        LLM instance for Ragas to use as judge.
                          If None and USE_HF_JUDGE=true, uses HuggingFaceEndpoint.
        embeddings:       Embeddings instance for AnswerRelevancy scoring.

    Returns:
        Dict with scalar float scores (0–1) for each metric:
          {
              "context_precision": 0.82,
              "answer_relevancy":  0.76,
              "faithfulness":      0.91,
          }

    Notes:
        - Ragas can fail on individual rows if the LLM returns malformed JSON.
          Such rows are skipped; the scores are averaged over successful rows.
        - If all rows fail, returns NaN scores rather than raising.
    """
    if not pipeline_outputs:
        logger.warning("run_ragas_eval called with empty pipeline_outputs")
        return {
            "context_precision": float("nan"),
            "answer_relevancy": float("nan"),
            "faithfulness": float("nan"),
        }

    # Build HuggingFace Dataset from pipeline outputs
    hf_dataset = Dataset.from_list(pipeline_outputs)

    judge_llm = _get_judge_llm(llm_judge)

    # Ragas v0.4+ requires llm at metric construction time
    metric_kwargs: dict = {}
    if judge_llm is not None:
        metric_kwargs["llm"] = judge_llm
    if embeddings is not None:
        metric_kwargs["embeddings"] = embeddings

    metrics = [
        ContextPrecision(**metric_kwargs),
        AnswerRelevancy(**metric_kwargs),
        Faithfulness(**metric_kwargs),
    ]

    eval_kwargs: dict = {"dataset": hf_dataset, "metrics": metrics}

    try:
        result = evaluate(**eval_kwargs)
        df = result.to_pandas()

        scores: dict[str, float] = {}
        for metric_name in ("context_precision", "answer_relevancy", "faithfulness"):
            if metric_name in df.columns:
                scores[metric_name] = float(df[metric_name].mean())
            else:
                scores[metric_name] = float("nan")

        return scores

    except Exception as exc:
        logger.error("Ragas evaluation failed: %s", exc, exc_info=True)
        return {
            "context_precision": float("nan"),
            "answer_relevancy": float("nan"),
            "faithfulness": float("nan"),
        }
