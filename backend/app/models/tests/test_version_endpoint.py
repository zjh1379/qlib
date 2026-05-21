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


async def test_version_returns_current_recorder(client):
    fake_versions = {
        "current": {
            "recorder_id": "abc123",
            "experiment": "rolling_v2_ensemble",
            "created_at": "2026-05-19T22:01:00",
            "metrics": {"ic_mean": 0.031, "ir": 2.6},
        },
        "previous": None,
        "previous_2": None,
        "next_retrain_at": "2026-05-24T22:00:00",
    }
    with patch("app.models.service.version_info", return_value=fake_versions):
        r = await client.get("/api/models/version")
    assert r.status_code == 200
    body = r.json()
    assert body["current"]["recorder_id"] == "abc123"
    assert body["current"]["metrics"]["ir"] == 2.6
