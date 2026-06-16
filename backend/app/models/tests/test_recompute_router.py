from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models.router import router
from app.models.schemas import RecomputeJob, RecomputeProgress, RecomputeTriggerResponse
import app.models.recompute as rc


def _client():
    app = FastAPI()
    app.include_router(router, prefix="/api/models")
    return TestClient(app)


def test_post_recompute_starts(monkeypatch):
    monkeypatch.setattr(rc, "trigger_recompute",
                        lambda view, models: RecomputeTriggerResponse(status="started", job_id="j1"))
    c = _client()
    r = c.post("/api/models/candidates/recompute", json={"view": "ensemble", "models": ["lgbm_5d"]})
    assert r.status_code == 200
    assert r.json() == {"status": "started", "job_id": "j1"}


def test_get_recompute_status(monkeypatch):
    job = RecomputeJob(job_id="j1", status="running", started_at="t", view="ensemble",
                       models=[], progress=RecomputeProgress(phase="metrics", percent=60, message="x"))
    monkeypatch.setattr(rc, "get_job", lambda jid: job if jid == "j1" else None)
    c = _client()
    assert c.get("/api/models/candidates/recompute/j1").json()["progress"]["percent"] == 60
    assert c.get("/api/models/candidates/recompute/nope").status_code == 404


def test_get_recompute_active(monkeypatch):
    monkeypatch.setattr(rc, "get_active_job", lambda: None)
    c = _client()
    r = c.get("/api/models/candidates/recompute/active")
    assert r.status_code == 200 and r.json() is None
