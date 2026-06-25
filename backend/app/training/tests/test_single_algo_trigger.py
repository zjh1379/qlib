import pytest
from httpx import ASGITransport, AsyncClient
from app.main import create_app
from app.scheduling.router import set_manager
from app.core.db import get_session


class _Mgr:
    def __init__(self): self.spec = "UNSET"
    async def run_now(self, session, force=False, run_spec=None):
        self.spec = run_spec
        return "jX"
    def get_job_status(self, jid): return None
    def get_active_job(self): return None


@pytest.mark.asyncio
async def test_single_algo_builds_reblend_spec():
    app = create_app(); m = _Mgr(); set_manager(m)
    app.dependency_overrides[get_session] = lambda: None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.post("/api/training/run", json={"scope": "single", "models": ["lgbm"], "force": True})
        assert r.status_code == 200 and r.json()["job_id"] == "jX"
    assert m.spec == ["reblend", "--only", "lgbm"]


@pytest.mark.asyncio
async def test_full_scope_uses_none_spec():
    app = create_app(); m = _Mgr(); set_manager(m)
    app.dependency_overrides[get_session] = lambda: None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        await ac.post("/api/training/run", json={"scope": "full", "force": True})
    assert m.spec is None
