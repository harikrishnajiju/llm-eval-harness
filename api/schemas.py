"""
Pydantic request/response schemas for the LLM Eval Harness API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class EvalRequest(BaseModel):
    """Request body for POST /runs."""

    model_name: str = Field(default="llama3", description="Ollama model tag to evaluate.")
    prompt_variant: Literal["default", "concise", "cot"] = Field(
        default="default",
        description="System prompt template to use. Controls what is A/B tested.",
    )
    n_samples: int = Field(
        default=25,
        ge=5,
        le=200,
        description="Number of SQuAD samples to evaluate. Must be between 5 and 200.",
    )


class RunStatus(BaseModel):
    """Minimal response returned immediately after POST /runs."""

    run_id: str
    status: str
    created_at: datetime


class RunDetail(BaseModel):
    """Full run record returned by GET /runs and GET /runs/{run_id}."""

    run_id: str
    model_name: str
    prompt_variant: str
    n_samples: int
    status: str
    metrics: dict[str, float] | None = None
    error: str | None = None
    duration_seconds: float | None = None
    created_at: datetime
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}
