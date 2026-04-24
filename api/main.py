"""
FastAPI application entry point.

Lifespan handles DB initialization on startup.
CORS is left open (no browser client in this project).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Load .env before anything else reads environment variables
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment
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
