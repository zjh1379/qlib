import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.ops.router import router as ops_router


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(ops_router, prefix="/api/ops", tags=["ops"])
    return app


@pytest.fixture
def client():
    app = _make_app()
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_health_ok(client):
    async with client as c:
        r = await c.get("/api/ops/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "qlib_ready" in body
