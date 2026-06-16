import time
import app.models.recompute as rc


def _wait_done(job_id, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = rc.get_job(job_id)
        if j and j.status in ("done", "failed"):
            return j
        time.sleep(0.01)
    raise AssertionError("job did not finish in time")


def test_trigger_runs_and_emits_progress(monkeypatch):
    def fake_candidates(top, days, min_top, experiment=None, view="ensemble", models=None):
        rc.emit_progress("metrics", 5, 10, "halfway")
        return {"items": []}
    monkeypatch.setattr("app.models.service.candidates", fake_candidates)

    resp = rc.trigger_recompute(view="ensemble", models=["lgbm_5d"])
    assert resp.status == "started" and resp.job_id
    job = _wait_done(resp.job_id)
    assert job.status == "done"
    assert job.progress.percent == 100  # final done overrides intermediate
    assert job.view == "ensemble" and job.models == ["lgbm_5d"]


def test_trigger_failure_sets_failed(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr("app.models.service.candidates", boom)
    resp = rc.trigger_recompute(view="alstm", models=[])
    job = _wait_done(resp.job_id)
    assert job.status == "failed"
    assert "kaboom" in (job.error or "")


def test_already_running_guard(monkeypatch):
    import threading
    gate = threading.Event()

    def slow(*a, **k):
        gate.wait(1.0)
        return {"items": []}
    monkeypatch.setattr("app.models.service.candidates", slow)
    r1 = rc.trigger_recompute(view="ensemble", models=[])
    r2 = rc.trigger_recompute(view="ensemble", models=[])
    assert r2.status == "already_running" and r2.job_id == r1.job_id
    gate.set()
    _wait_done(r1.job_id)
