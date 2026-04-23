"""
FastAPI route definitions for the LLM Eval Harness.

Endpoints:
  POST  /runs           — Trigger a new eval run (async background task)
  GET   /runs           — List recent runs
  GET   /runs/{run_id}  — Get a specific run's status & metrics
  GET   /health         — Health check (no auth required)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import EvalRequest, RunDetail, RunStatus
from storage.database import get_run, list_runs, save_run
from storage.models import EvalRun

logger = logging.getLogger(__name__)

router = APIRouter()

_API_KEY = os.getenv("API_KEY", "dev-secret-key")
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


async def verify_api_key(x_api_key: str = Header(...)) -> None:
    """Reject requests whose X-API-Key header does not match the configured key."""
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Background eval task
# ---------------------------------------------------------------------------


async def _run_eval(run_id: str, request: EvalRequest) -> None:
    """
    Full eval pipeline executed as a FastAPI BackgroundTask.

    Flow:
      1. Load dataset
      2. Build embeddings + FAISS vector store
      3. Build RAG chain
      4. Run pipeline
      5. Score with Ragas
      6. Persist results to SQLite
    """
    start = time.monotonic()

    try:
        # --- Phase 1: Load dataset ---
        from data_loader.hf_dataset import load_qa_dataset

        dataset_rows = load_qa_dataset(n_samples=request.n_samples)

        # --- Phase 2: RAG pipeline ---
        from rag_pipeline.embeddings import get_embeddings
        from rag_pipeline.vectorstore import build_vectorstore
        from rag_pipeline.chain import build_rag_chain, run_pipeline

        embeddings = get_embeddings()
        contexts = [row["context"] for row in dataset_rows]
        vectorstore = build_vectorstore(contexts, embeddings)

        chain = build_rag_chain(
            vectorstore=vectorstore,
            prompt_variant=request.prompt_variant,
            model_name=request.model_name,
            ollama_base_url=_OLLAMA_BASE_URL,
        )

        pipeline_outputs = run_pipeline(chain, dataset_rows)

        # --- Phase 3: Ragas evaluation ---
        from evaluator.ragas_eval import run_ragas_eval
        from langchain_ollama import ChatOllama  # type: ignore

        llm_judge = ChatOllama(model=request.model_name, base_url=_OLLAMA_BASE_URL)
        metrics = run_ragas_eval(pipeline_outputs, llm_judge=llm_judge, embeddings=embeddings)

        duration = time.monotonic() - start

        # --- Phase 4: Persist results ---
        run = await get_run(run_id)
        if run is None:
            logger.error("Run %s not found in DB after completion", run_id)
            return

        run.status = "complete"
        run.metrics = json.dumps(metrics)
        run.duration_seconds = duration
        run.completed_at = datetime.now(tz=timezone.utc)
        await save_run(run)

        logger.info("Run %s completed in %.1fs. Metrics: %s", run_id, duration, metrics)

    except Exception as exc:
        duration = time.monotonic() - start
        logger.error("Run %s failed: %s", run_id, exc, exc_info=True)

        run = await get_run(run_id)
        if run:
            run.status = "failed"
            run.error = str(exc)
            run.duration_seconds = duration
            run.completed_at = datetime.now(tz=timezone.utc)
            await save_run(run)


# ---------------------------------------------------------------------------
# Helper: ORM → schema
# ---------------------------------------------------------------------------


def _orm_to_detail(run: EvalRun) -> RunDetail:
    metrics = None
    if run.metrics:
        try:
            metrics = json.loads(run.metrics)
        except json.JSONDecodeError:
            metrics = None

    return RunDetail(
        run_id=run.id,
        model_name=run.model_name,
        prompt_variant=run.prompt_variant,
        n_samples=run.n_samples,
        status=run.status,
        metrics=metrics,
        error=run.error,
        duration_seconds=run.duration_seconds,
        created_at=run.created_at,
        completed_at=run.completed_at,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/runs", response_model=RunStatus, status_code=202)
async def create_run(
    request: EvalRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_api_key),
) -> RunStatus:
    """Trigger a new eval run. Returns immediately with the run_id."""
    run_id = str(uuid.uuid4())
    now = datetime.now(tz=timezone.utc)

    run = EvalRun(
        id=run_id,
        model_name=request.model_name,
        prompt_variant=request.prompt_variant,
        n_samples=request.n_samples,
        status="running",
        created_at=now,
    )
    await save_run(run)

    background_tasks.add_task(_run_eval, run_id, request)

    return RunStatus(run_id=run_id, status="running", created_at=now)


@router.get("/runs", response_model=list[RunDetail])
async def get_runs(_: None = Depends(verify_api_key)) -> list[RunDetail]:
    """Return the 50 most recent eval runs."""
    runs = await list_runs(limit=50)
    return [_orm_to_detail(r) for r in runs]


@router.get("/runs/{run_id}", response_model=RunDetail)
async def get_run_detail(
    run_id: str,
    _: None = Depends(verify_api_key),
) -> RunDetail:
    """Return full details for a single eval run."""
    run = await get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return _orm_to_detail(run)


@router.get("/health")
async def health() -> JSONResponse:
    """
    Health check endpoint — no auth required.
    Pings Ollama to report connectivity status.
    """
    ollama_reachable = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{_OLLAMA_BASE_URL}/api/tags")
            ollama_reachable = resp.status_code == 200
    except Exception:
        ollama_reachable = False

    return JSONResponse({"status": "ok", "ollama_reachable": ollama_reachable})
