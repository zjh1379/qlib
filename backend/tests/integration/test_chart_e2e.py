import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_chart_real_data_for_maotai(client):
    r = await client.get(
        "/api/charts/SH600519",
        params={"start": "2025-01-01", "end": "2026-05-08", "with_pred": "true"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "SH600519"
    actual = body["actual"]
    predicted = body["predicted"]
    assert len(actual) > 200
    assert len(predicted) > 0
    # last actual should match start of forecast window
    last_actual = actual[-1]["time"]
    assert body["meta"]["last_actual_date"] == last_actual


@pytest.mark.asyncio
async def test_chart_alignment_invariant(client):
    """Property: every predicted bar's open == prior actual bar's close (within float epsilon)."""
    r = await client.get(
        "/api/charts/SH600519",
        params={"start": "2025-04-01", "end": "2026-05-08", "with_pred": "true"},
    )
    body = r.json()
    actual_by_time = {b["time"]: b for b in body["actual"]}
    actual_times = sorted(actual_by_time.keys())
    for pb in body["predicted"]:
        idx = actual_times.index(pb["time"])
        prior = actual_by_time[actual_times[idx - 1]]
        assert abs(pb["open"] - prior["close"]) < 1e-6
