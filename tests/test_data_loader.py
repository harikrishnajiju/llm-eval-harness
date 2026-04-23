"""
Tests for data_loader/hf_dataset.py.

Mocks datasets.load_dataset so these tests never hit the network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_fake_dataset(n: int):
    """Build a minimal mock that looks like a HuggingFace SQuAD dataset."""
    rows = [
        {
            "question": f"Question {i}?",
            "context": f"Context passage {i}.",
            "answers": {"text": [f"Answer {i}"], "answer_start": [0]},
        }
        for i in range(n)
    ]

    class _FakeDataset:
        def __init__(self, data):
            self._data = data

        def __len__(self):
            return len(self._data)

        def __iter__(self):
            return iter(self._data)

        def shuffle(self, seed=None):
            return self

        def select(self, indices):
            return _FakeDataset([self._data[i] for i in indices])

    return _FakeDataset(rows)


@patch("data_loader.hf_dataset.load_dataset")
def test_load_returns_correct_keys(mock_load):
    """Each returned row must have the three expected keys."""
    mock_load.return_value = _make_fake_dataset(10)

    from data_loader.hf_dataset import load_qa_dataset

    rows = load_qa_dataset(n_samples=5)

    assert len(rows) == 5
    for row in rows:
        assert "question" in row
        assert "context" in row
        assert "ground_truth" in row


@patch("data_loader.hf_dataset.load_dataset")
def test_load_respects_n_samples(mock_load):
    """n_samples must cap the output even when the dataset is larger."""
    mock_load.return_value = _make_fake_dataset(100)

    from data_loader.hf_dataset import load_qa_dataset

    rows = load_qa_dataset(n_samples=10)
    assert len(rows) == 10


@patch("data_loader.hf_dataset.load_dataset")
def test_load_values_are_strings(mock_load):
    """All three fields must be non-empty strings."""
    mock_load.return_value = _make_fake_dataset(5)

    from data_loader.hf_dataset import load_qa_dataset

    rows = load_qa_dataset(n_samples=5)

    for row in rows:
        assert isinstance(row["question"], str) and row["question"]
        assert isinstance(row["context"], str) and row["context"]
        assert isinstance(row["ground_truth"], str) and row["ground_truth"]


@patch("data_loader.hf_dataset.load_dataset")
def test_load_calls_with_trust_remote_code_false(mock_load):
    """Must call load_dataset with trust_remote_code=False."""
    mock_load.return_value = _make_fake_dataset(5)

    from data_loader.hf_dataset import load_qa_dataset

    load_qa_dataset()

    _, kwargs = mock_load.call_args
    assert kwargs.get("trust_remote_code") is False
