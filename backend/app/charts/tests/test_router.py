import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from app.charts.router import router as charts_router
from app.core.exceptions import BusinessError


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(charts_router, prefix="/api/charts", tags=["charts"])

    @app.exception_handler(BusinessError)
    async def biz_handler(_, exc: BusinessError):
        return JSONResponse(status_code=exc.http_status, content=exc.as_response_dict())

    return app


@pytest.fixture
async def client():
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_get_chart_200(client):
    r = await client.get(
        "/api/charts/SH600519",
        params={"start": "2025-04-01", "end": "2026-05-08", "with_pred": "true"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "SH600519"
    assert isinstance(body["actual"], list)
    assert len(body["actual"]) > 5
    assert isinstance(body["predicted"], list)


@pytest.mark.asyncio
async def test_get_chart_unknown_symbol_404(client):
    r = await client.get(
        "/api/charts/SH999999",
        params={"start": "2025-04-01", "end": "2025-05-01", "with_pred": "false"},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["code"] in {"ohlcv_empty", "symbol_missing"}
