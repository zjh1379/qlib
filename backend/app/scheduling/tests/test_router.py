import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        async with app.router.lifespan_context(app):
            yield c


async def test_get_schedule_returns_default(client):
    r = await client.get("/api/scheduling/retrain")
    assert r.status_code == 200
    body = r.json()
    assert body["day_of_week"] == 6
    assert body["hour"] == 22
    assert body["minute"] == 0
    assert body["enabled"] is True


async def test_put_schedule_updates_row(client):
    r = await client.put(
        "/api/scheduling/retrain",
        json={"day_of_week": 5, "hour": 23, "minute": 30, "enabled": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["day_of_week"] == 5
    assert body["hour"] == 23
    assert body["minute"] == 30


async def test_put_schedule_rejects_trading_hours(client):
    r = await client.put(
        "/api/scheduling/retrain",
        json={"day_of_week": 1, "hour": 10, "minute": 0, "enabled": True},  # Tue 10:00
    )
    assert r.status_code == 400
    # Project convention: BusinessError.as_response_dict() emits "code", not
    # "error_code". The spec asserts on the error_code key but the actual
    # global handler emits "code" — adjusted to match the existing handler.
    assert "trading_hours" in r.json()["code"]
