from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        async with app.router.lifespan_context(app):
            yield c


async def test_predictions_returns_base_scores_per_day(client):
    fake_history = {
        "symbol": "SH600000",
        "name": "浦发银行",
        "experiment": "rolling_v2_ensemble",
        "points": [
            {
                "date": "2026-05-15",
                "score": 0.12,
                "rank": 5,
                "universe_size": 800,
                "base_scores": {
                    "lgbm_1d": 0.10, "lgbm_5d": 0.13, "lgbm_20d": 0.14,
                    "alstm_1d": 0.08, "alstm_5d": 0.11, "alstm_20d": 0.15,
                    "tra_1d": 0.09, "tra_5d": 0.12, "tra_20d": 0.13,
                },
            },
        ],
    }
    with patch("app.models.service.prediction_history", return_value=fake_history):
        r = await client.get("/api/models/predictions/SH600000?view=alstm")
    assert r.status_code == 200
    body = r.json()
    assert body["points"][0]["base_scores"]["alstm_5d"] == 0.11


async def test_predictions_default_view_is_ensemble(client):
    fake_history = {
        "symbol": "SH600000", "name": "", "experiment": "rolling_v2_ensemble", "points": [],
    }
    with patch("app.models.service.prediction_history", return_value=fake_history) as mock_svc:
        r = await client.get("/api/models/predictions/SH600000")
    assert r.status_code == 200
    assert mock_svc.call_args.kwargs.get("view", "ensemble") == "ensemble"
