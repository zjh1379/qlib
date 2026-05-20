"""Tests for the P2 portfolio backend module.

Each test runs against an isolated SQLite DB (tmp_path) and a freshly-built
FastAPI app that includes only the portfolio router. The qlib_adapter calls
made inside `get_holdings` are stubbed so tests don't depend on real qlib data.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.db import Base, dispose_db_singletons, init_db_singletons
from app.core.exceptions import BusinessError
from app.portfolio import orm as _orm_module  # noqa: F401 -- registers the ORM table
from app.portfolio import service as portfolio_service
from app.portfolio.router import router as portfolio_router


def _create_tables_sync(db_url: str) -> None:
    """Create all ORM tables on a fresh engine (sync wrapper)."""
    from sqlalchemy.ext.asyncio import create_async_engine

    async def _create() -> None:
        engine = create_async_engine(db_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(_create())


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("QLIB_COMPANION_APP_DB_PATH", str(db_path))

    settings = Settings()
    _create_tables_sync(settings.db_url)
    init_db_singletons(settings)

    # Stub out qlib-dependent helpers so tests don't require real qlib data.
    monkeypatch.setattr(
        portfolio_service, "_build_name_lookup", lambda: {"SH600000": "Test Stock"}
    )
    monkeypatch.setattr(
        portfolio_service, "_fetch_current_prices", lambda symbols: ({}, None)
    )

    app = FastAPI()
    app.include_router(portfolio_router, prefix="/api/portfolio")

    @app.exception_handler(BusinessError)
    async def biz_handler(_request, exc: BusinessError):
        return JSONResponse(status_code=exc.http_status, content=exc.as_response_dict())

    with TestClient(app) as c:
        yield c

    asyncio.run(dispose_db_singletons())


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_add_transaction_creates_holding(client):
    r = client.post(
        "/api/portfolio/transactions",
        json={
            "symbol": "SH600000",
            "kind": "buy",
            "qty": 100,
            "price": 10.0,
            "fee": 5.0,
            "executed_at": "2026-05-01T09:30:00",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["symbol"] == "SH600000"
    assert body["qty"] == 100
    assert body["id"] >= 1

    r2 = client.get("/api/portfolio/holdings")
    assert r2.status_code == 200, r2.text
    h_body = r2.json()
    assert len(h_body["holdings"]) == 1
    h = h_body["holdings"][0]
    assert h["symbol"] == "SH600000"
    assert h["qty"] == 100
    # avg_cost = (100*10 + 5) / 100 = 10.05
    assert h["avg_cost"] == pytest.approx(10.05)
    # effective_cost = (1005 - 0) / 100 = 10.05
    assert h["effective_cost"] == pytest.approx(10.05)
    # name populated from stub
    assert h["name"] == "Test Stock"


def test_partial_sell_reduces_qty(client):
    client.post(
        "/api/portfolio/transactions",
        json={
            "symbol": "SH600000",
            "kind": "buy",
            "qty": 100,
            "price": 10.0,
            "executed_at": "2026-05-01T09:30:00",
        },
    )
    r = client.post(
        "/api/portfolio/transactions",
        json={
            "symbol": "SH600000",
            "kind": "sell",
            "qty": 30,
            "price": 12.0,
            "executed_at": "2026-05-05T09:30:00",
        },
    )
    assert r.status_code == 200, r.text

    h = client.get("/api/portfolio/holdings").json()
    assert len(h["holdings"]) == 1
    assert h["holdings"][0]["qty"] == pytest.approx(70)


def test_full_sell_removes_holding(client):
    client.post(
        "/api/portfolio/transactions",
        json={
            "symbol": "SH600000",
            "kind": "buy",
            "qty": 100,
            "price": 10.0,
            "executed_at": "2026-05-01T09:30:00",
        },
    )
    r = client.post(
        "/api/portfolio/transactions",
        json={
            "symbol": "SH600000",
            "kind": "sell",
            "qty": 100,
            "price": 12.0,
            "executed_at": "2026-05-05T09:30:00",
        },
    )
    assert r.status_code == 200, r.text

    h = client.get("/api/portfolio/holdings").json()
    assert h["holdings"] == []


def test_avg_cost_weighted(client):
    client.post(
        "/api/portfolio/transactions",
        json={
            "symbol": "SH600000",
            "kind": "buy",
            "qty": 100,
            "price": 10.0,
            "executed_at": "2026-05-01T09:30:00",
        },
    )
    client.post(
        "/api/portfolio/transactions",
        json={
            "symbol": "SH600000",
            "kind": "buy",
            "qty": 100,
            "price": 20.0,
            "executed_at": "2026-05-02T09:30:00",
        },
    )

    h = client.get("/api/portfolio/holdings").json()
    assert len(h["holdings"]) == 1
    holding = h["holdings"][0]
    assert holding["qty"] == pytest.approx(200)
    # avg_cost = (100*10 + 100*20) / 200 = 15  (fees both zero)
    assert holding["avg_cost"] == pytest.approx(15.0)
    assert holding["effective_cost"] == pytest.approx(15.0)


def test_delete_transaction_reverts_state(client):
    r1 = client.post(
        "/api/portfolio/transactions",
        json={
            "symbol": "SH600000",
            "kind": "buy",
            "qty": 100,
            "price": 10.0,
            "executed_at": "2026-05-01T09:30:00",
        },
    )
    r2 = client.post(
        "/api/portfolio/transactions",
        json={
            "symbol": "SH600000",
            "kind": "buy",
            "qty": 50,
            "price": 12.0,
            "executed_at": "2026-05-02T09:30:00",
        },
    )
    tx2_id = r2.json()["id"]

    # Before delete: qty 150
    h_before = client.get("/api/portfolio/holdings").json()
    assert h_before["holdings"][0]["qty"] == pytest.approx(150)

    # Delete the second buy
    d = client.delete(f"/api/portfolio/transactions/{tx2_id}")
    assert d.status_code == 204, d.text

    h_after = client.get("/api/portfolio/holdings").json()
    assert h_after["holdings"][0]["qty"] == pytest.approx(100)
    assert h_after["holdings"][0]["avg_cost"] == pytest.approx(10.0)


def test_validation_rejects_short_sell(client):
    r = client.post(
        "/api/portfolio/transactions",
        json={
            "symbol": "SH600000",
            "kind": "sell",
            "qty": 10,
            "price": 12.0,
            "executed_at": "2026-05-01T09:30:00",
        },
    )
    # ConflictError -> 409
    assert r.status_code in (400, 409), r.text
    body = r.json()
    assert body["code"] == "oversell"


def test_unknown_transaction_404(client):
    r = client.get("/api/portfolio/transactions/99999")
    assert r.status_code == 404, r.text
    body = r.json()
    assert body["code"] == "transaction_missing"
