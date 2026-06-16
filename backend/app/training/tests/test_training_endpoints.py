import pytest
from httpx import ASGITransport, AsyncClient

from app.core.db import get_session
from app.main import create_app
from app.scheduling.router import set_manager


class _FakeManager:
    def __init__(self):
        self._entry = None

    async def run_now(self, session, force=False):
        self._entry = {
            "job_id": "tjob1", "kind": "manual", "status": "running",
            "started_at": "s", "finished_at": None, "error": None, "log_path": None,
        }
        return "tjob1"

    def get_job_status(self, job_id):
        if self._entry and self._entry["job_id"] == job_id:
            return self._entry
        return None

    def get_active_job(self):
        return self._entry


@pytest.mark.asyncio
async def test_run_then_status_and_active(monkeypatch):
    app = create_app()
    # create_app() does NOT run the lifespan, so DB singletons are not
    # initialized and get_session would raise. The faked run_now ignores the
    # session, so override the dependency to a no-op.
    app.dependency_overrides[get_session] = lambda: None
    fake = _FakeManager()
    set_manager(fake)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/api/training/run", json={"scope": "full", "force": True})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        assert job_id == "tjob1"

        r2 = await ac.get(f"/api/training/jobs/{job_id}")
        assert r2.status_code == 200
        assert r2.json()["status"] == "running"

        r3 = await ac.get("/api/training/jobs/active")
        assert r3.status_code == 200
        assert r3.json()["job_id"] == "tjob1"


@pytest.mark.asyncio
async def test_status_unknown_job_returns_null():
    app = create_app()
    app.dependency_overrides[get_session] = lambda: None
    set_manager(_FakeManager())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/training/jobs/nope")
        assert r.status_code == 200
        assert r.json() is None
