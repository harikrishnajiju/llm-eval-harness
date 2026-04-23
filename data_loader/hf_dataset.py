"""
Load and slice a HuggingFace QA dataset (default: rajpurkar/squad).

Returns clean (question, context, ground_truth) dicts ready for the RAG pipeline.
"""

from __future__ import annotations

from datasets import load_dataset  # type: ignore


def load_qa_dataset(
    dataset_name: str = "rajpurkar/squad",
    split: str = "validation",
    n_samples: int = 25,
    seed: int = 42,
) -> list[dict]:
    """
    Load a QA dataset from HuggingFace and return a sliced, shuffled list.

    Each returned dict has:
      - question:     str
      - context:      str
      - ground_truth: str  (first answer from SQuAD answers field)

    Args:
        dataset_name: HuggingFace dataset identifier.
        split:        Dataset split to use (usually "validation" for SQuAD).
        n_samples:    Number of samples to return.
        seed:         Random seed for reproducible shuffling.

    Returns:
        List of dicts with keys: question, context, ground_truth.
    """
    ds = load_dataset(dataset_name, split=split, trust_remote_code=False)

    # Shuffle for reproducibility, then take the first n_samples
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(min(n_samples, len(ds))))

    rows: list[dict] = []
    for item in ds:
        # SQuAD answers field: {"text": [...], "answer_start": [...]}
        ground_truth = item["answers"]["text"][0]
        rows.append(
            {
                "question": item["question"],
                "context": item["context"],
                "ground_truth": ground_truth,
            }
        )

    return rows
