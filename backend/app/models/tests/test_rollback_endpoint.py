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


async def test_rollback_to_previous_archives_current(client):
    fake_result = {
        "status": "rolled_back",
        "archived_recorder_id": "current_xyz",
        "new_current_recorder_id": "prev_abc",
        "reason": None,
    }
    with patch("app.models.service.rollback_to", return_value=fake_result):
        r = await client.post("/api/models/rollback", json={"target": "previous_1"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "rolled_back"
    assert body["archived_recorder_id"] == "current_xyz"


async def test_rollback_with_no_history_is_noop(client):
    fake_result = {
        "status": "no_op",
        "archived_recorder_id": None,
        "new_current_recorder_id": None,
        "reason": "no_previous_recorder",
    }
    with patch("app.models.service.rollback_to", return_value=fake_result):
        r = await client.post("/api/models/rollback", json={"target": "previous_1"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "no_op"
    assert body["reason"] == "no_previous_recorder"
