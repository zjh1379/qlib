from pathlib import Path

import pytest

from app.scheduling.service import SchedulerManager


@pytest.mark.asyncio
async def test_run_now_sets_log_path_and_invokes_job_with_it(tmp_path, monkeypatch):
    import app.core.db as _db
    monkeypatch.setattr(_db, "_session_maker", None)
    captured = {}

    async def fake_job(job_id: str, log_path: Path, run_spec=None) -> None:
        captured["job_id"] = job_id
        captured["log_path"] = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text('PROGRESS {"phase":"done","current":1,"total":1,"message":"ok"}\n', encoding="utf-8")

    mgr = SchedulerManager(fake_job, logs_dir=tmp_path)
    # Bypass APScheduler: drive _gated_job_fn directly with a pre-registered entry.
    job_id = "retrain_weekly_manual_test"
    mgr._remember_job(job_id, {
        "job_id": job_id, "kind": "manual", "status": "pending",
        "started_at": None, "finished_at": None, "queued_at": "t", "error": None,
        "log_path": str(tmp_path / f"api_retrain_{job_id}.log"),
    })
    mgr._active_job_id = job_id
    await mgr._gated_job_fn(_tracked_job_id=job_id)

    entry = mgr.get_job_status(job_id)
    assert entry["status"] == "done"
    assert entry["log_path"] == str(tmp_path / f"api_retrain_{job_id}.log")
    assert captured["job_id"] == job_id
    assert captured["log_path"] == tmp_path / f"api_retrain_{job_id}.log"


@pytest.mark.asyncio
async def test_job_raising_marks_failed(tmp_path, monkeypatch):
    import app.core.db as _db
    monkeypatch.setattr(_db, "_session_maker", None)

    async def boom(job_id: str, log_path: Path, run_spec=None) -> None:
        raise RuntimeError("rolling_train exited 1")

    mgr = SchedulerManager(boom, logs_dir=tmp_path)
    job_id = "retrain_weekly_manual_boom"
    mgr._remember_job(job_id, {
        "job_id": job_id, "kind": "manual", "status": "pending",
        "started_at": None, "finished_at": None, "queued_at": "t", "error": None,
        "log_path": str(tmp_path / f"api_retrain_{job_id}.log"),
    })
    with pytest.raises(RuntimeError):
        await mgr._gated_job_fn(_tracked_job_id=job_id)
    assert mgr.get_job_status(job_id)["status"] == "failed"
    assert "exited 1" in mgr.get_job_status(job_id)["error"]
