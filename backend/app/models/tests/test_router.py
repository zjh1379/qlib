import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.core.exceptions import BusinessError
from app.models.router import router as models_router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(models_router, prefix="/api/models")

    @app.exception_handler(BusinessError)
    async def biz_handler(_, exc):
        return JSONResponse(status_code=exc.http_status, content=exc.as_response_dict())

    return TestClient(app)


def test_screen_default_returns_top30(client):
    r = client.get("/api/models/screen")
    assert r.status_code == 200
    body = r.json()
    assert body["window_days"] == 5
    assert len(body["items"]) == 30
    assert body["items"][0]["rank"] == 1
    # score_avg should be non-increasing
    scores = [it["score_avg"] for it in body["items"]]
    assert scores == sorted(scores, reverse=True)


def test_screen_min_top_filters(client):
    # ask for very strict filter — fewer items should come back
    r = client.get("/api/models/screen?top=30&days=5&min_top=4")
    assert r.status_code == 200
    body = r.json()
    for it in body["items"]:
        assert it["days_in_top"] >= 4


def test_screen_custom_top(client):
    r = client.get("/api/models/screen?top=10")
    body = r.json()
    assert len(body["items"]) == 10


def test_predictions_for_known_symbol(client):
    r = client.get("/api/models/predictions/SH600519?days=30")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "SH600519"
    assert len(body["points"]) > 10
    pt = body["points"][0]
    assert "date" in pt and "score" in pt and "rank" in pt and "universe_size" in pt


def test_predictions_unknown_symbol_404(client):
    r = client.get("/api/models/predictions/SH999999")
    assert r.status_code == 404
    assert r.json()["code"] == "symbol_missing"


def test_experiments_lists_at_least_one(client):
    r = client.get("/api/models/experiments")
    assert r.status_code == 200
    body = r.json()
    # daily_cn_fresh should be present in this env
    names = [e["name"] for e in body["experiments"]]
    assert any("daily_cn_fresh" in n or n == "daily_cn_fresh" for n in names)
