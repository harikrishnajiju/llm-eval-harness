"""
Integration tests for the FastAPI API layer.

Uses an in-memory SQLite DB and mocked pipeline to test all endpoints.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Health endpoint (no auth)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint(test_client):
    """GET /health should return 200 with status=ok."""
    with patch("api.routes.httpx.AsyncClient") as mock_http:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_http.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_resp
        )
        resp = await test_client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "ollama_reachable" in data


@pytest.mark.asyncio
async def test_health_no_auth_required(test_client):
    """GET /health must succeed even without the X-API-Key header."""
    from httpx import ASGITransport, AsyncClient

    from api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/health")

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_api_key_returns_403(test_client):
    """
    POST /runs without X-API-Key returns 422 (FastAPI validates required
    headers before the auth dependency runs) or 403.
    Either is an acceptable rejection.
    """
    from httpx import ASGITransport, AsyncClient

    from api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/runs", json={"n_samples": 5})

    # FastAPI raises 422 for missing required Header fields before auth runs
    assert resp.status_code in (403, 422)


@pytest.mark.asyncio
async def test_wrong_api_key_returns_403(test_client):
    """POST /runs with wrong key must return 403."""
    from httpx import ASGITransport, AsyncClient

    from api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"X-API-Key": "wrong-key"},
    ) as client:
        resp = await client.post("/runs", json={"n_samples": 5})

    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# POST /runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_run_returns_run_id(test_client):
    """POST /runs must return a run_id and status=running."""
    resp = await test_client.post(
        "/runs",
        json={"model_name": "llama3", "prompt_variant": "default", "n_samples": 5},
    )

    assert resp.status_code == 202
    data = resp.json()
    assert "run_id" in data
    assert data["status"] == "running"
    assert "created_at" in data


@pytest.mark.asyncio
async def test_create_run_invalid_variant(test_client):
    """POST /runs with invalid prompt_variant must return 422."""
    resp = await test_client.post(
        "/runs",
        json={"prompt_variant": "invalid_variant", "n_samples": 5},
    )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_run_n_samples_out_of_range(test_client):
    """POST /runs with n_samples < 5 or > 200 must return 422."""
    resp = await test_client.post("/runs", json={"n_samples": 1})
    assert resp.status_code == 422

    resp = await test_client.post("/runs", json={"n_samples": 500})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /runs and GET /runs/{run_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_runs_empty(mock_ollama, mock_ragas_evaluate, mock_embeddings):
    """
    GET /runs on a brand-new in-memory DB should return an empty list.
    Uses a dedicated client with a fresh engine to avoid cross-test contamination.
    """
    import os
    from httpx import ASGITransport, AsyncClient
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import storage.database as db_module

    # Spin up a completely isolated in-memory engine for this test
    fresh_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    fresh_session = async_sessionmaker(fresh_engine, expire_on_commit=False)
    original_engine = db_module.engine
    original_session = db_module.AsyncSessionFactory

    db_module.engine = fresh_engine
    db_module.AsyncSessionFactory = fresh_session

    try:
        from storage.models import Base
        async with fresh_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        from api.main import app
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"X-API-Key": "test-api-key"},
        ) as client:
            resp = await client.get("/runs")

        assert resp.status_code == 200
        assert resp.json() == []
    finally:
        db_module.engine = original_engine
        db_module.AsyncSessionFactory = original_session
        await fresh_engine.dispose()


@pytest.mark.asyncio
async def test_get_run_not_found(test_client):
    """GET /runs/{id} for a non-existent run must return 404."""
    resp = await test_client.get("/runs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_then_get_run(test_client):
    """Create a run then fetch it by ID — should be retrievable."""
    post_resp = await test_client.post(
        "/runs",
        json={"model_name": "llama3", "prompt_variant": "concise", "n_samples": 5},
    )
    assert post_resp.status_code == 202
    run_id = post_resp.json()["run_id"]

    get_resp = await test_client.get(f"/runs/{run_id}")
    assert get_resp.status_code == 200

    data = get_resp.json()
    assert data["run_id"] == run_id
    assert data["prompt_variant"] == "concise"


@pytest.mark.asyncio
async def test_create_then_list_runs(test_client):
    """After creating a run, GET /runs must include it."""
    post_resp = await test_client.post(
        "/runs",
        json={"model_name": "llama3", "prompt_variant": "cot", "n_samples": 5},
    )
    run_id = post_resp.json()["run_id"]

    list_resp = await test_client.get("/runs")
    assert list_resp.status_code == 200

    run_ids = [r["run_id"] for r in list_resp.json()]
    assert run_id in run_ids
