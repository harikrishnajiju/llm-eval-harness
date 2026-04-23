"""
Tests for evaluator/ragas_eval.py.

Mocks ragas.evaluate AND the metric constructors so no LLM calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_pipeline_outputs(n: int = 3) -> list[dict]:
    return [
        {
            "question": f"Question {i}?",
            "answer": f"Answer {i}.",
            "contexts": [f"Context {i}."],
            "ground_truth": f"Ground truth {i}.",
        }
        for i in range(n)
    ]


def _make_fake_result(scores: dict):
    class _FakeResult:
        def to_pandas(self_inner):
            return pd.DataFrame([scores])

    return _FakeResult()


# Patch both evaluate() and the metric classes (they require an llm arg in v0.4)
_METRIC_PATCH_TARGETS = [
    "evaluator.ragas_eval.ContextPrecision",
    "evaluator.ragas_eval.AnswerRelevancy",
    "evaluator.ragas_eval.Faithfulness",
]


@patch("evaluator.ragas_eval.evaluate")
@patch("evaluator.ragas_eval.Faithfulness")
@patch("evaluator.ragas_eval.AnswerRelevancy")
@patch("evaluator.ragas_eval.ContextPrecision")
def test_run_ragas_eval_returns_three_metrics(mock_cp, mock_ar, mock_f, mock_eval):
    """Result must contain all three required metric keys."""
    mock_eval.return_value = _make_fake_result(
        {
            "context_precision": 0.8,
            "answer_relevancy": 0.75,
            "faithfulness": 0.9,
        }
    )

    from evaluator.ragas_eval import run_ragas_eval

    scores = run_ragas_eval(_make_pipeline_outputs(), llm_judge=MagicMock())

    assert set(scores.keys()) == {"context_precision", "answer_relevancy", "faithfulness"}


@patch("evaluator.ragas_eval.evaluate")
@patch("evaluator.ragas_eval.Faithfulness")
@patch("evaluator.ragas_eval.AnswerRelevancy")
@patch("evaluator.ragas_eval.ContextPrecision")
def test_run_ragas_eval_scores_in_range(mock_cp, mock_ar, mock_f, mock_eval):
    """All scores must be floats between 0 and 1."""
    mock_eval.return_value = _make_fake_result(
        {
            "context_precision": 0.82,
            "answer_relevancy": 0.76,
            "faithfulness": 0.91,
        }
    )

    from evaluator.ragas_eval import run_ragas_eval

    scores = run_ragas_eval(_make_pipeline_outputs(), llm_judge=MagicMock())

    for key, val in scores.items():
        assert isinstance(val, float), f"{key} should be float"
        assert 0.0 <= val <= 1.0, f"{key}={val} out of range"


def test_run_ragas_eval_empty_input():
    """Empty pipeline_outputs should return NaN scores without crashing."""
    from evaluator.ragas_eval import run_ragas_eval
    import math

    scores = run_ragas_eval([])

    for val in scores.values():
        assert math.isnan(val)


@patch("evaluator.ragas_eval.evaluate")
@patch("evaluator.ragas_eval.Faithfulness")
@patch("evaluator.ragas_eval.AnswerRelevancy")
@patch("evaluator.ragas_eval.ContextPrecision")
def test_run_ragas_eval_graceful_on_exception(mock_cp, mock_ar, mock_f, mock_eval):
    """If ragas.evaluate raises, return NaN scores rather than propagating."""
    import math

    mock_eval.side_effect = RuntimeError("LLM returned malformed JSON")

    from evaluator.ragas_eval import run_ragas_eval

    scores = run_ragas_eval(_make_pipeline_outputs(), llm_judge=MagicMock())

    for val in scores.values():
        assert math.isnan(val)
