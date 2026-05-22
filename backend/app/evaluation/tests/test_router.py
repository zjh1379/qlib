import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        async with app.router.lifespan_context(app):
            yield c


async def test_list_recorders_endpoint(client):
    r = await client.get("/api/evaluation/recorders")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    # We expect at least daily_cn_fresh in the dev env
    experiments = {row["experiment"] for row in body}
    assert "daily_cn_fresh" in experiments


async def test_run_evaluation_endpoint(client):
    # First get the daily_cn_fresh recorder id
    r = await client.get("/api/evaluation/recorders")
    recorders = r.json()
    target = next((row for row in recorders if row["experiment"] == "daily_cn_fresh"), None)
    if target is None:
        pytest.skip("no daily_cn_fresh recorder")

    payload = {
        "recorder_id": target["recorder_id"],
        "top_k": 30,
        "cost_bps": 10,
        "force_refresh": False,
    }
    r = await client.post("/api/evaluation/run", json=payload, timeout=120)
    assert r.status_code == 200
    body = r.json()
    assert body["recorder_id"] == target["recorder_id"]
    assert "scorecard" in body
    assert "acceptance" in body
    assert "regimes" in body
    assert body["acceptance"]["details"].keys() >= {
        "ic_mean", "ir", "max_drawdown", "daily_turnover", "regimes_all_positive",
    }


async def test_run_evaluation_unknown_recorder_returns_404(client):
    payload = {"recorder_id": "nonexistent_recorder_xxx", "top_k": 30, "cost_bps": 10}
    r = await client.post("/api/evaluation/run", json=payload)
    assert r.status_code == 404


async def test_get_results_endpoint_after_run(client):
    r = await client.get("/api/evaluation/recorders")
    target = next(
        (row for row in r.json() if row["experiment"] == "daily_cn_fresh"), None
    )
    if target is None:
        pytest.skip("no daily_cn_fresh recorder")

    # Make sure it's evaluated
    await client.post(
        "/api/evaluation/run",
        json={"recorder_id": target["recorder_id"], "top_k": 30, "cost_bps": 10},
        timeout=120,
    )

    r = await client.get(f"/api/evaluation/results/{target['recorder_id']}")
    assert r.status_code == 200
    assert r.json()["recorder_id"] == target["recorder_id"]


async def test_get_results_unknown_returns_404(client):
    r = await client.get("/api/evaluation/results/nonexistent_xxx")
    assert r.status_code == 404


async def test_compare_endpoint_same_recorder(client):
    r = await client.get("/api/evaluation/recorders")
    target = next(
        (row for row in r.json() if row["experiment"] == "daily_cn_fresh"), None
    )
    if target is None:
        pytest.skip("no daily_cn_fresh recorder")

    rid = target["recorder_id"]
    r = await client.get(f"/api/evaluation/compare?a={rid}&b={rid}", timeout=120)
    assert r.status_code == 200
    body = r.json()
    assert abs(body["ic_delta"]) < 1e-9
    assert body["verdict"] == "no significant difference"
