import time
from app.analysis import service
from app.analysis.schemas import AiAnalysis


def test_disabled_returns_disabled(monkeypatch):
    monkeypatch.setattr(service, "_is_enabled", lambda s: False)
    resp = service.trigger_analysis(reason="manual_ui")
    assert resp.status == "disabled"
    assert service.get_active_job() is None


def test_trigger_runs_worker_and_persists(monkeypatch, tmp_path):
    db = tmp_path / "app.db"
    monkeypatch.setattr(service, "_is_enabled", lambda s: True)
    monkeypatch.setattr(service, "_db_path", lambda s: str(db))
    monkeypatch.setattr(service, "_top_n", lambda s: 2)
    monkeypatch.setattr(service, "_load_picks", lambda: (
        "2026-06-10",
        [("SH600519", "贵州茅台", {"score_today": 0.9}),
         ("SZ000001", "平安银行", {"score_today": 0.8})],
    ))
    def fake_one(symbol, name, ctx, as_of):
        return AiAnalysis(interpretation=f"n-{symbol}", stance="neutral",
                          model="test-model", as_of_date=as_of, status="ok")
    monkeypatch.setattr(service, "_analyze_symbol", fake_one)
    captured = {}
    monkeypatch.setattr(service.store, "upsert_many",
                        lambda path, rows: captured.update({"n": len(rows), "path": path}))

    resp = service.trigger_analysis(reason="manual_ui")
    assert resp.status == "started"
    for _ in range(50):
        job = service.get_job(resp.job_id)
        if job and job.status == "done":
            break
        time.sleep(0.05)
    job = service.get_job(resp.job_id)
    assert job.status == "done"
    assert job.analyzed == 2
    assert captured["n"] == 2
    assert service.get_status().is_running is False


def test_double_trigger_is_already_running(monkeypatch):
    import threading
    monkeypatch.setattr(service, "_is_enabled", lambda s: True)
    gate = threading.Event()
    monkeypatch.setattr(service, "_run_picks", lambda *a, **k: gate.wait(2))
    r1 = service.trigger_analysis(reason="manual_ui")
    r2 = service.trigger_analysis(reason="manual_ui")
    gate.set()
    assert r1.status == "started"
    assert r2.status == "already_running"
