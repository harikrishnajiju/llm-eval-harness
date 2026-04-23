"""
SQLAlchemy ORM model for eval run records.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class EvalRun(Base):
    """
    Represents a single evaluation run.

    status transitions:  "running" → "complete" | "failed"
    metrics:  JSON string, e.g. '{"context_precision": 0.82, ...}'
              NULL while running, populated on completion.
    """

    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    prompt_variant: Mapped[str] = mapped_column(String, nullable=False)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    metrics: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
