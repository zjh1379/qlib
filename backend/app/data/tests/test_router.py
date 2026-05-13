from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from app.core.exceptions import BusinessError
from app.data.router import instruments_router, router as data_router


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(data_router, prefix="/api/data", tags=["data"])
    app.include_router(instruments_router, prefix="/api", tags=["data"])

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
async def test_data_status_200(client):
    r = await client.get("/api/data/status")
    assert r.status_code == 200, r.text
    body = r.json()
    # required fields present
    for key in ("calendar_end", "calendar_size", "instruments_count", "last_refresh_at", "freshness"):
        assert key in body
    assert isinstance(body["calendar_size"], int)
    assert body["calendar_size"] > 0
    assert isinstance(body["instruments_count"], int)
    assert body["instruments_count"] > 0
    assert body["freshness"] in {"fresh", "stale_1d", "stale_2d_plus"}
    # calendar_end ISO date YYYY-MM-DD
    assert len(body["calendar_end"]) == 10 and body["calendar_end"][4] == "-"


@pytest.mark.asyncio
async def test_instruments_csi300(client):
    r = await client.get("/api/instruments", params={"market": "csi300"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["market"] == "csi300"
    assert isinstance(body["items"], list)
    assert len(body["items"]) >= 250
    assert body["count"] == len(body["items"])
    for item in body["items"][:5]:
        assert "symbol" in item and "name" in item
    # at least one item has a non-empty Chinese name (production cache populated)
    assert any(item["name"] for item in body["items"])


@pytest.mark.asyncio
async def test_instruments_unknown_market_400(client):
    r = await client.get("/api/instruments", params={"market": "foo"})
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["code"] == "unsupported_market"
    assert body["context"]["market"] == "foo"


@pytest.mark.asyncio
async def test_refresh_starts_job(client):
    """Mock subprocess.Popen so we never actually launch the Python refresh script."""
    # Reset module state to avoid contention with other tests
    from app.data import service as data_service

    data_service._refresh_jobs.clear()
    data_service._running_job_id = None
    # Make sure lock is released (defensive — should already be released)
    try:
        data_service._running_lock.release()
    except RuntimeError:
        pass

    fake_proc = MagicMock()
    fake_proc.pid = 12345
    # wait() returns 0 (success). The thread will release the lock after this returns.
    fake_proc.wait.return_value = 0

    with patch("app.data.service.subprocess.Popen", return_value=fake_proc) as mock_popen:
        r = await client.post("/api/data/refresh")
        assert r.status_code == 200, r.text
        body = r.json()
        assert "job_id" in body and body["job_id"]
        assert "started_at" in body
        assert "message" in body

        # Verify Popen was called: [<python>, <path_to>/production/incremental_refresh.py]
        assert mock_popen.called
        call_args, call_kwargs = mock_popen.call_args
        cmd = call_args[0]
        assert len(cmd) == 2
        # second arg is the script path
        assert cmd[1].replace("\\", "/").endswith("production/incremental_refresh.py")

    # Allow the background thread to finish so subsequent tests can acquire the lock
    import time

    for _ in range(20):
        if data_service._running_job_id is None:
            break
        time.sleep(0.05)


@pytest.mark.asyncio
async def test_refresh_status_unknown_job_404(client):
    r = await client.get("/api/data/refresh/nonexistent")
    assert r.status_code == 404, r.text
    body = r.json()
    assert body["code"] == "job_missing"
