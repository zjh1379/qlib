import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.core.db import get_session


@pytest.mark.asyncio
async def test_runs_endpoint_merges_runs_and_recorders(monkeypatch):
    from app.training import service as svc

    class _Run:
        def __init__(self, **k): self.__dict__.update(k)
    runs = [
        _Run(job_id="j1", kind="manual", scope="full", models_json=None, status="done",
             started_at="t0", finished_at="t1", recorder_id="recA", error=None, created_at="2026-06-16T02:00:00"),
        _Run(job_id="j2", kind="manual", scope="full", models_json=None, status="failed",
             started_at="t2", finished_at="t3", recorder_id=None, error="boom", created_at="2026-06-16T01:00:00"),
    ]

    async def fake_list_runs(session, limit=100):
        return runs

    class _Rec:
        def __init__(self, **k): self.__dict__.update(k)
    recs = [
        _Rec(recorder_id="recA", experiment="rolling_v2_ensemble", run_name="ensemble_2026-06-16",
             created_at="2026-06-16T02:00:00", pred_start="2026-01-01", pred_end="2026-06-16",
             pred_rows=1000, has_eval=True, ic_mean=0.04, ir=2.1, acceptance_passed=True),
        _Rec(recorder_id="recOld", experiment="rolling_v2_ensemble", run_name="ensemble_2026-06-01",
             created_at="2026-06-01T02:00:00", pred_start="2026-01-01", pred_end="2026-06-01",
             pred_rows=900, has_eval=False, ic_mean=None, ir=None, acceptance_passed=None),
    ]
    monkeypatch.setattr(svc, "_list_runs", fake_list_runs, raising=False)
    monkeypatch.setattr(svc, "list_recorders_with_summary", lambda: recs, raising=False)

    app = create_app()
    app.dependency_overrides[get_session] = lambda: None
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/training/runs")
    assert r.status_code == 200
    rows = r.json()
    by_id = {row.get("recorder_id") or row["job_id"]: row for row in rows}
    assert by_id["recA"]["status"] == "done"
    assert by_id["recA"]["ic_mean"] == 0.04
    assert by_id["j2"]["status"] == "failed" and by_id["j2"]["recorder_id"] is None
    assert by_id["recOld"]["status"] == "historical"


@pytest.mark.asyncio
async def test_runs_endpoint_marks_candidates(monkeypatch):
    from app.training import service as svc

    async def fake_list_runs(session, limit=100):
        return []

    class _Rec:
        def __init__(self, **k): self.__dict__.update(k)
    recs = [
        _Rec(recorder_id="recCand", experiment="rolling_v2_ensemble_candidates",
             run_name="candidate_lgbm_2026-06-16", created_at="2026-06-16T03:00:00",
             pred_start=None, pred_end=None, pred_rows=None, has_eval=False,
             ic_mean=None, ir=None, acceptance_passed=None),
        _Rec(recorder_id="recProd", experiment="rolling_v2_ensemble",
             run_name="ensemble_2026-06-16", created_at="2026-06-16T02:00:00",
             pred_start=None, pred_end=None, pred_rows=None, has_eval=True,
             ic_mean=0.04, ir=2.0, acceptance_passed=True),
    ]
    monkeypatch.setattr(svc, "_list_runs", fake_list_runs, raising=False)
    monkeypatch.setattr(svc, "list_recorders_with_summary", lambda: recs, raising=False)

    app = create_app()
    app.dependency_overrides[get_session] = lambda: None
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/training/runs")
    rows = {row["recorder_id"]: row for row in r.json()}
    assert rows["recCand"]["is_candidate"] is True
    assert rows["recCand"]["experiment"] == "rolling_v2_ensemble_candidates"
    assert rows["recProd"]["is_candidate"] is False
