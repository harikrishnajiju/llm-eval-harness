"""
FastAPI application entry point.

Lifespan handles DB initialization on startup.
CORS is left open (no browser client in this project).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from storage.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup/shutdown logic around the application lifecycle."""
    logger.info("Starting LLM Eval Harness — initializing database…")
    await init_db()
    logger.info("Database ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="LLM Eval Harness",
    description=(
        "A self-hosted evaluation harness for RAG pipelines. "
        "Trigger eval runs, compare prompt variants, and inspect Ragas metrics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
