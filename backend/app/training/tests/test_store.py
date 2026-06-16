import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession

from app.core.db import Base
from app.training.orm import TrainingRunORM
from app.training import store


@pytest_asyncio.fixture
async def sm(tmp_path):
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path/'t.db'}")
    async with eng.begin() as conn:
        await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=[TrainingRunORM.__table__]))
    yield async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)
    await eng.dispose()


@pytest.mark.asyncio
async def test_record_start_then_finish_and_list(sm):
    async with sm() as s:
        await store.record_run_start(s, job_id="j1", kind="manual", scope="full", models=None, started_at="t0")
    async with sm() as s:
        await store.record_run_finish(s, job_id="j1", status="done", recorder_id="rec123", error=None, finished_at="t1")
    async with sm() as s:
        runs = await store.list_runs(s)
    assert len(runs) == 1
    assert runs[0].job_id == "j1"
    assert runs[0].status == "done"
    assert runs[0].recorder_id == "rec123"
    assert runs[0].finished_at == "t1"


@pytest.mark.asyncio
async def test_finish_unknown_job_is_noop(sm):
    async with sm() as s:
        await store.record_run_finish(s, job_id="ghost", status="failed", recorder_id=None, error="x", finished_at="t")
    async with sm() as s:
        assert await store.list_runs(s) == []
