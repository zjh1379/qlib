from pathlib import Path

import pytest

import app.scheduling.service as svc
from app.scheduling.service import make_subprocess_retrain_job


class _EmptyStdout:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _FakeProc:
    def __init__(self):
        self.stdout = _EmptyStdout()

    async def wait(self):
        return 0


@pytest.mark.asyncio
async def test_job_argv_full_vs_reblend(monkeypatch, tmp_path):
    captured = []

    async def fake_exec(*args, **kwargs):
        captured.append(args)
        return _FakeProc()

    monkeypatch.setattr(svc.asyncio, "create_subprocess_exec", fake_exec)
    job = make_subprocess_retrain_job(python_path="py", repo_root=tmp_path)

    # run_spec=None → full retrain ("run-once")
    await job("j1", tmp_path / "a.log", None)
    # run_spec set → single-algo re-blend ("reblend --only lgbm")
    await job("j2", tmp_path / "b.log", ["reblend", "--only", "lgbm"])

    assert "run-once" in captured[0]
    assert "reblend" in captured[1] and "lgbm" in captured[1]
