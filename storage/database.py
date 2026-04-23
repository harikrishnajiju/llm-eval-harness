"""
Async SQLite database engine, session factory, and CRUD helpers.

Uses SQLAlchemy async engine + aiosqlite so FastAPI async endpoints
never block the event loop on database calls.
"""

from __future__ import annotations

import os

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from storage.models import Base, EvalRun

# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

_DB_PATH = os.getenv("DB_PATH", "./data/evals.db")

# For in-memory testing pass DB_PATH=":memory:"
if _DB_PATH == ":memory:":
    _DATABASE_URL = "sqlite+aiosqlite:///:memory:"
else:
    # Ensure parent directory exists at import time (best effort)
    _db_dir = os.path.dirname(_DB_PATH)
    if _db_dir:
        os.makedirs(_db_dir, exist_ok=True)
    _DATABASE_URL = f"sqlite+aiosqlite:///{_DB_PATH}"

engine = create_async_engine(
    _DATABASE_URL,
    echo=False,
    # SQLite-specific: allow use across threads (aiosqlite handles this)
    connect_args={"check_same_thread": False},
)

AsyncSessionFactory = async_sessionmaker(engine, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """Create all tables if they do not yet exist. Called on FastAPI startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------


async def save_run(run: EvalRun) -> None:
    """Insert or update (merge) a run record."""
    async with AsyncSessionFactory() as session:
        async with session.begin():
            await session.merge(run)


async def get_run(run_id: str) -> EvalRun | None:
    """Fetch a single EvalRun by its ID. Returns None if not found."""
    async with AsyncSessionFactory() as session:
        result = await session.execute(select(EvalRun).where(EvalRun.id == run_id))
        return result.scalar_one_or_none()


async def list_runs(limit: int = 50) -> list[EvalRun]:
    """Return the most recent runs, ordered by created_at DESC."""
    async with AsyncSessionFactory() as session:
        result = await session.execute(
            select(EvalRun).order_by(EvalRun.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
