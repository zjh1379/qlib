from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.analysis.router import router, internal_router
from app.analysis import service
from app.analysis.schemas import TriggerResponse, AnalysisStatus, AnalysisJob


def _client():
    app = FastAPI()
    app.include_router(router)
    app.include_router(internal_router)
    return TestClient(app)


def test_run_now_delegates(monkeypatch):
    monkeypatch.setattr(service, "trigger_analysis",
                        lambda reason="manual_ui": TriggerResponse(status="started", job_id="j1"))
    r = _client().post("/api/analysis/run-now")
    assert r.status_code == 200
    assert r.json()["status"] == "started"


def test_status_endpoint(monkeypatch):
    monkeypatch.setattr(service, "get_status",
                        lambda: AnalysisStatus(is_running=True, last_run_at="t"))
    r = _client().get("/api/analysis/status")
    assert r.json()["is_running"] is True


def test_get_job_404(monkeypatch):
    monkeypatch.setattr(service, "get_job", lambda jid: None)
    r = _client().get("/api/analysis/jobs/nope")
    assert r.status_code == 404


def test_active_peek(monkeypatch):
    monkeypatch.setattr(service, "get_active_job",
                        lambda: AnalysisJob(job_id="j2", status="running", started_at="t"))
    r = _client().get("/api/analysis/active/peek")
    assert r.json()["job_id"] == "j2"


def test_internal_refresh_testclient_allowed(monkeypatch):
    monkeypatch.setattr(service, "trigger_analysis",
                        lambda reason="data_refresh": TriggerResponse(status="started", job_id="j3"))
    r = _client().post("/api/internal/analysis/refresh")
    assert r.status_code == 200
    assert r.json()["status"] == "started"
