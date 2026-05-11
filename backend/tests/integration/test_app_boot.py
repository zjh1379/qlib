import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_app_health(client):
    r = await client.get("/api/ops/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_openapi_docs_accessible(client):
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    paths = spec["paths"]
    assert "/api/charts/{symbol}" in paths
    assert "/api/ops/health" in paths


@pytest.mark.asyncio
async def test_business_error_returns_proper_status(client):
    r = await client.get(
        "/api/charts/SH999999",
        params={"start": "2025-01-01", "end": "2025-02-01"},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["code"] in {"ohlcv_empty", "symbol_missing"}
    assert "detail" in body
