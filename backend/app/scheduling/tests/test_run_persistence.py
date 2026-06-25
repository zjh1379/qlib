from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession

from app.core.db import Base
from app.training.orm import TrainingRunORM
from app.training import store
from app.scheduling.service import SchedulerManager
import app.core.db as _db


@pytest_asyncio.fixture
async def wired_db(tmp_path, monkeypatch):
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path/'t.db'}")
    async with eng.begin() as conn:
        await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=[TrainingRunORM.__table__]))
    sm = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(_db, "_session_maker", sm)
    yield sm
    await eng.dispose()


@pytest.mark.asyncio
async def test_job_lifecycle_writes_training_run(tmp_path, wired_db):
    async def fake_job(job_id: str, log_path: Path, run_spec=None, profile_name: str = "conservative") -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("PROGRESS {\"phase\":\"done\",\"current\":1,\"total\":1,\"message\":\"\"}\nRECORDER rec_ok\n", encoding="utf-8")

    mgr = SchedulerManager(fake_job, logs_dir=tmp_path)
    job_id = "retrain_weekly_manual_persist"
    mgr._remember_job(job_id, {
        "job_id": job_id, "kind": "manual", "status": "pending",
        "started_at": None, "finished_at": None, "queued_at": "t", "error": None,
        "log_path": str(tmp_path / f"api_retrain_{job_id}.log"),
    })
    mgr._active_job_id = job_id
    await mgr._gated_job_fn(_tracked_job_id=job_id)

    async with wired_db() as s:
        runs = await store.list_runs(s)
    assert len(runs) == 1
    assert runs[0].job_id == job_id
    assert runs[0].status == "done"
    assert runs[0].recorder_id == "rec_ok"
    assert runs[0].finished_at is not None
