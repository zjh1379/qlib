# 训练工作台 P2 — 历史 + 评估 + 对比 + 回滚 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Fold model history, evaluation, multi-version compare, and rollback into the 训练工作台 page, and persist every training run (incl. failed/in-progress) in a `training_runs` table linked to its produced recorder.

**Architecture:** Reuse the existing backend wherever it exists — recorder history (`evaluation.list_recorders_with_summary`), 8-metric evaluation + paired-t compare (`/api/evaluation/{recorders,run,results,compare}`), and rollback (`/api/models/{version,rollback}`). NEW: a `training_runs` sqlite table (alembic 0005) populated by the `SchedulerManager` job lifecycle; `rolling_train` emits its recorder id (`RECORDER <id>` stdout line) which the backend captures from the per-job log; a `GET /api/training/runs` endpoint that unions training_runs (all attempts) with recorder summaries (metrics); and ③历史 + ④对比 sections added to `Training.tsx`.

**Tech Stack:** FastAPI / SQLAlchemy async + Alembic (backend), React + TanStack Query + Tailwind + lightweight-charts (frontend), pytest + vitest.

**Scope decisions (confirmed with user):**
- **Promote = existing rollback** (revert to previous recorder). Arbitrary "promote any candidate to current" is DEFERRED to P3 — the serve layer picks current = newest recorder with a loadable `pred.pkl` (`qlib_adapter.get_latest_recorder_id`), and a true candidate/promote needs a current-pointer mechanism that only matters once single-algo experimental models exist (P3).
- **Per-epoch loss curves DEFERRED to P3** (needs qlib pytorch-model instrumentation).
- P2 runs are always `scope="full"` (single-algo is P3); `training_runs.scope/models` columns are recorded now but only become non-trivial in P3.

**Out of scope for P2:** single-algo training, model registry, loss curves, arbitrary-candidate promote. Do NOT build these.

---

## Key reuse points (verified)
- `backend/app/evaluation/service.py:29` `list_recorders_with_summary() -> list[RecorderSummary]` — all trained models with `recorder_id, experiment, run_name, created_at, pred_start/end, pred_rows, has_eval, ic_mean, ir, acceptance_passed`.
- `backend/app/evaluation/router.py` — `GET /recorders`, `POST /run`, `GET /results/{id}`, `GET /compare?a&b` (→ `CompareResult`).
- `backend/app/models/service.py:742` `version_info()`, `:794` `rollback_to(target)`; `backend/app/models/router.py:85` `GET /version`, `:90` `POST /rollback`.
- `backend/app/core/qlib_adapter.py:366` `get_latest_recorder_id` = current selection (newest start_time with loadable pred.pkl).
- `production/rolling_train.py` ensemble recorder block (`with R.start(experiment_name=cfg.experiment_name, recorder_name=f"ensemble_{end_date}"):`, ~line 538).
- Alembic: `backend/alembic/versions/0004_add_ai_analysis_table.py` (template); ORM base `backend/app/core/db.py:14`; ad-hoc async session `async with app.core.db._session_maker() as session:`.
- Frontend reuse: `api.evaluation.{listRecorders,run,getResult,compare}` + `api.models.{version,rollback}` (client.ts), `frontend/src/pages/evaluation/{CompareCard,Scorecard,AcceptanceLights,RecorderRow}.tsx`, `frontend/src/pages/evaluation/hooks.ts` (`useRecorders,useCompare,useRunEvaluation`), `frontend/src/jobs/toast.tsx`, `frontend/src/portfolio/HoldingsTable.tsx` (Th/Td table pattern), multi-select FIFO from `Evaluation.tsx:15-20`.

---

## Task 1: `training_runs` ORM + alembic 0005

**Files:**
- Create: `backend/app/training/orm.py`
- Create: `backend/alembic/versions/0005_add_training_runs.py`
- Test: `backend/app/training/tests/test_orm_migration.py`

**Worktree:** all work in `E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a` (branch `feat/training-studio`); full paths + `git -C`. Python `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8`; backend cmds with cwd = `...\backend`.

- [ ] **Step 1: Write the failing test**

```python
# backend/app/training/tests/test_orm_migration.py
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from app.training.orm import TrainingRunORM
from app.core.db import Base


def test_training_run_orm_creates_and_roundtrips(tmp_path: Path):
    # Build the table from metadata and round-trip a row.
    from sqlalchemy.orm import Session
    eng = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(eng, tables=[TrainingRunORM.__table__])
    with Session(eng) as s:
        s.add(TrainingRunORM(
            job_id="j1", kind="manual", scope="full", models_json=None,
            status="running", started_at="2026-06-16T01:00:00", finished_at=None,
            recorder_id=None, error=None,
        ))
        s.commit()
        row = s.get(TrainingRunORM, "j1")
        assert row.status == "running"
        assert row.scope == "full"
        assert row.recorder_id is None


def test_table_name_is_training_runs():
    assert TrainingRunORM.__tablename__ == "training_runs"
```

- [ ] **Step 2: Run test to verify it fails**

Run (cwd `backend`): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/tests/test_orm_migration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.training.orm'`

- [ ] **Step 3: Create the ORM**

```python
# backend/app/training/orm.py
from sqlalchemy import Column, DateTime, String
from sqlalchemy.sql import func

from app.core.db import Base


class TrainingRunORM(Base):
    """One row per training run attempt (manual or cron). Populated by the
    SchedulerManager job lifecycle. recorder_id is filled on success once the
    training subprocess emits its `RECORDER <id>` line."""

    __tablename__ = "training_runs"

    job_id = Column(String, primary_key=True)
    kind = Column(String, nullable=False, default="manual")          # cron | manual
    scope = Column(String, nullable=False, default="full")           # full (P2) | single (P3)
    models_json = Column(String, nullable=True)                      # JSON list for single-algo (P3); null = all
    status = Column(String, nullable=False, default="pending")       # pending|running|done|failed|skipped
    started_at = Column(String, nullable=True)
    finished_at = Column(String, nullable=True)
    recorder_id = Column(String, nullable=True)
    error = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
```

- [ ] **Step 4: Create the alembic migration**

```python
# backend/alembic/versions/0005_add_training_runs.py
"""add training_runs table

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-16

Training studio P2: one row per training run attempt (manual/cron), linked to
its produced recorder once known.
"""
from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_runs",
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False, server_default="manual"),
        sa.Column("scope", sa.String(), nullable=False, server_default="full"),
        sa.Column("models_json", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("started_at", sa.String(), nullable=True),
        sa.Column("finished_at", sa.String(), nullable=True),
        sa.Column("recorder_id", sa.String(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.current_timestamp()),
        sa.PrimaryKeyConstraint("job_id"),
    )


def downgrade() -> None:
    op.drop_table("training_runs")
```

- [ ] **Step 5: Run test + apply migration**

Run (cwd `backend`): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/tests/test_orm_migration.py -v` → PASS (2 passed).
Then apply the migration to the dev DB: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m alembic upgrade head` (run from `backend`; expects "Running upgrade 0004 -> 0005"). If alembic isn't directly invokable that way, match how the project runs migrations (check `backend/alembic.ini` / any `make`/script). Report the actual output.

- [ ] **Step 6: Commit**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add backend/app/training/orm.py backend/alembic/versions/0005_add_training_runs.py backend/app/training/tests/test_orm_migration.py
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "feat(training): training_runs ORM + alembic 0005"
```

---

## Task 2: `training/store.py` — async run persistence

**Files:**
- Create: `backend/app/training/store.py`
- Test: `backend/app/training/tests/test_store.py`

**Context:** Async ORM writes (the SchedulerManager job runs in the asyncio loop, so we use async sessions directly — not the sync-sqlite3 trick the analysis slice uses for its threadpool job). `list_runs` reads newest-first.

- [ ] **Step 1: Write the failing test**

```python
# backend/app/training/tests/test_store.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run (cwd `backend`): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.training.store'` (and confirm `pytest_asyncio` is available; the project already uses async tests — match its fixture style if different).

- [ ] **Step 3: Implement the store**

```python
# backend/app/training/store.py
"""training_runs persistence (async). The SchedulerManager job lifecycle writes
a row on start and updates it on finish; the history endpoint reads via list_runs."""
from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.training.orm import TrainingRunORM


async def record_run_start(
    session: AsyncSession, *, job_id: str, kind: str, scope: str,
    models: list[str] | None, started_at: str,
) -> None:
    """Insert (or replace) a run row in 'running' state."""
    existing = await session.get(TrainingRunORM, job_id)
    if existing is None:
        session.add(TrainingRunORM(
            job_id=job_id, kind=kind, scope=scope,
            models_json=json.dumps(models) if models else None,
            status="running", started_at=started_at,
        ))
    else:
        existing.status = "running"
        existing.started_at = started_at
    await session.commit()


async def record_run_finish(
    session: AsyncSession, *, job_id: str, status: str,
    recorder_id: str | None, error: str | None, finished_at: str,
) -> None:
    """Update a run row's terminal state. No-op if the job_id is unknown."""
    row = await session.get(TrainingRunORM, job_id)
    if row is None:
        return
    row.status = status
    row.recorder_id = recorder_id
    row.error = error
    row.finished_at = finished_at
    await session.commit()


async def list_runs(session: AsyncSession, limit: int = 100) -> list[TrainingRunORM]:
    """Return run rows, newest first (by created_at)."""
    res = await session.execute(
        select(TrainingRunORM).order_by(TrainingRunORM.created_at.desc()).limit(limit)
    )
    return list(res.scalars().all())
```

- [ ] **Step 4: Run to verify it passes**

Run (cwd `backend`): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/tests/test_store.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add backend/app/training/store.py backend/app/training/tests/test_store.py
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "feat(training): training_runs async store (record start/finish + list)"
```

---

## Task 3: rolling_train emits its recorder id

**Files:**
- Modify: `production/progress.py` (add `emit_recorder`)
- Modify: `production/rolling_train.py` (call it inside the ensemble recorder block)
- Test: `production/tests/test_progress_emit.py` (extend)

**Context:** The backend needs the produced recorder id to link a run → recorder. The ensemble recorder is created at `with R.start(experiment_name=cfg.experiment_name, recorder_name=f"ensemble_{end_date}"):` (~line 538 of rolling_train.py). Inside that block, `R.get_recorder().id` is the id. Emit it as a `RECORDER <id>` stdout line (same capture path as PROGRESS).

- [ ] **Step 1: Write the failing test (extend existing file)**

Add to `production/tests/test_progress_emit.py`:

```python
def test_emit_recorder_prints_recorder_line(capsys):
    from production.progress import emit_recorder
    emit_recorder("abc123def")
    out = capsys.readouterr().out.strip()
    assert out == "RECORDER abc123def"
```

- [ ] **Step 2: Run to verify it fails**

Run (repo root): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest production/tests/test_progress_emit.py::test_emit_recorder_prints_recorder_line -v`
Expected: FAIL — `ImportError: cannot import name 'emit_recorder'`

- [ ] **Step 3: Add `emit_recorder` to progress.py**

Append to `production/progress.py`:

```python
def emit_recorder(recorder_id: str) -> None:
    """Emit the produced recorder id so the backend can link a training run to
    its recorder (parsed from the per-job log alongside PROGRESS lines)."""
    print(f"RECORDER {recorder_id}", flush=True)
```

- [ ] **Step 4: Call it in rolling_train.py**

In `production/rolling_train.py`, inside the ensemble recorder block (the `with R.start(experiment_name=cfg.experiment_name, recorder_name=f"ensemble_{end_date}"):` body, after the `if is_live_fold(end_date): R.save_objects(...)` / metrics logging), add an emit. Locate by the anchor `recorder_name=f"ensemble_{end_date}"`. Add near the top of the `with` body:

```python
            try:
                emit_recorder(R.get_recorder().id)
            except Exception:
                pass
```

(`emit_recorder` is already importable — extend the existing `from production.progress import emit_progress` import to `from production.progress import emit_progress, emit_recorder`.)

- [ ] **Step 5: Run to verify the emit test passes + no regression**

Run (repo root): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest production/tests/test_progress_emit.py production/tests/test_progress_total.py -v`
Expected: PASS (all). The rolling_train edit is a guarded one-liner inside an existing try-wrapped block; it cannot change run_once's return.

- [ ] **Step 6: Commit**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add production/progress.py production/rolling_train.py production/tests/test_progress_emit.py
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "feat(training): rolling_train emits RECORDER <id> for run-to-recorder linkage"
```

---

## Task 4: `latest_recorder_id` log parser

**Files:**
- Modify: `backend/app/training/service.py` (add `latest_recorder_id`)
- Test: `backend/app/training/tests/test_progress_parse.py` (extend)

- [ ] **Step 1: Write the failing test (extend existing file)**

Add to `backend/app/training/tests/test_progress_parse.py`:

```python
def test_latest_recorder_id_parses_recorder_line(tmp_path):
    from app.training.service import latest_recorder_id
    log = tmp_path / "j.log"
    log.write_text(
        "PROGRESS {\"phase\":\"done\",\"current\":6,\"total\":6,\"message\":\"done\"}\n"
        "RECORDER rec_abc\n",
        encoding="utf-8",
    )
    assert latest_recorder_id(log) == "rec_abc"


def test_latest_recorder_id_none_when_absent(tmp_path):
    from app.training.service import latest_recorder_id
    log = tmp_path / "j.log"
    log.write_text("no recorder here\n", encoding="utf-8")
    assert latest_recorder_id(log) is None
    assert latest_recorder_id(tmp_path / "missing.log") is None
```

- [ ] **Step 2: Run to verify it fails**

Run (cwd `backend`): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/tests/test_progress_parse.py -v`
Expected: FAIL — `ImportError: cannot import name 'latest_recorder_id'`

- [ ] **Step 3: Implement the parser**

Add to `backend/app/training/service.py`:

```python
def latest_recorder_id(log_path: Path) -> str | None:
    """Return the recorder id from the last 'RECORDER <id>' line, or None."""
    if not log_path.is_file():
        return None
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 32 * 1024)
            f.seek(size - read_size)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("RECORDER "):
                rid = line[len("RECORDER "):].strip()
                return rid or None
        return None
    except Exception:
        return None
```

- [ ] **Step 4: Run to verify it passes**

Run (cwd `backend`): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/tests/test_progress_parse.py -v`
Expected: PASS (6 passed — 4 prior + 2 new)

- [ ] **Step 5: Commit**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add backend/app/training/service.py backend/app/training/tests/test_progress_parse.py
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "feat(training): latest_recorder_id log parser"
```

---

## Task 5: Wire training_runs into the SchedulerManager job lifecycle

**Files:**
- Modify: `backend/app/scheduling/service.py` (`_gated_job_fn`)
- Test: `backend/app/scheduling/tests/test_run_persistence.py`

**Context:** When a job flips to `running`, write a `training_runs` row (kind from the entry, scope="full", models=None for P2). On terminal state, update it with status + recorder_id (parsed from the job's log) + error + finished_at. Use an ad-hoc async session from `app.core.db._session_maker`. To avoid an import cycle (scheduling ↔ training), import `app.training.store` / `app.training.service` LAZILY inside the method. DB failures must never crash the job (wrap in try/except + log).

- [ ] **Step 1: Write the failing test**

```python
# backend/app/scheduling/tests/test_run_persistence.py
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
    async def fake_job(job_id: str, log_path: Path) -> None:
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
```

- [ ] **Step 2: Run to verify it fails**

Run (cwd `backend`): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/scheduling/tests/test_run_persistence.py -v`
Expected: FAIL — the run row is not written (no persistence wired yet).

- [ ] **Step 3: Add the persistence hooks to `_gated_job_fn`**

In `backend/app/scheduling/service.py`, add a small helper method on `SchedulerManager` and call it at the lifecycle transitions. Add this method:

```python
    async def _persist_run(self, *, job_id: str, phase: str, entry: dict | None) -> None:
        """Best-effort write to training_runs. Never raises (DB optional)."""
        from app.core import db as _db
        if _db._session_maker is None:
            return
        try:
            from datetime import datetime as _dt
            now = _dt.now(tz=_CST).isoformat()
            async with _db._session_maker() as session:
                if phase == "start":
                    from app.training import store
                    await store.record_run_start(
                        session, job_id=job_id,
                        kind=(entry or {}).get("kind", "manual"),
                        scope="full", models=None, started_at=now,
                    )
                else:
                    from app.training import store
                    from app.training.service import latest_recorder_id
                    log_path = (entry or {}).get("log_path")
                    rid = latest_recorder_id(Path(log_path)) if log_path else None
                    await store.record_run_finish(
                        session, job_id=job_id, status=phase,
                        recorder_id=rid, error=(entry or {}).get("error"),
                        finished_at=now,
                    )
        except Exception:
            _log.warning("persist_training_run_failed", job_id=job_id, phase=phase, exc_info=True)
```

Then call it inside `_gated_job_fn`, at these points (the existing status-transition sites):
- When status becomes `running` (right after `entry["status"] = "running"; entry["started_at"] = ...`): `await self._persist_run(job_id=_tracked_job_id, phase="start", entry=entry)`
- On success (after `entry["status"] = "done"`): `await self._persist_run(job_id=_tracked_job_id, phase="done", entry=entry)`
- On failure (in the `except Exception` branch, after `entry["error"] = str(exc)`): `await self._persist_run(job_id=_tracked_job_id, phase="failed", entry=entry)` — call it BEFORE the `raise`.
- (Skip the `skipped` branch — a skipped job did nothing.)

Locate these by the existing `entry["status"] = ...` assignments in `_gated_job_fn`. The finish persistence must run AFTER `entry["error"]`/`entry["finished_at"]` are set so the recorder log is complete; placing the `done` call right after `entry["status"]="done"` is fine (the subprocess already finished + flushed its RECORDER line by then).

- [ ] **Step 4: Run to verify it passes**

Run (cwd `backend`): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/scheduling/tests/test_run_persistence.py app/scheduling/ -v`
Expected: PASS (new test + all prior scheduling tests still green).

- [ ] **Step 5: Commit**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add backend/app/scheduling/service.py backend/app/scheduling/tests/test_run_persistence.py
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "feat(training): SchedulerManager persists training_runs on job start/finish (best-effort)"
```

---

## Task 6: `GET /api/training/runs` — history (runs ∪ recorders)

**Files:**
- Modify: `backend/app/training/schemas.py` (add `TrainingRunRow`)
- Modify: `backend/app/training/service.py` (add `build_history`)
- Modify: `backend/app/training/router.py` (add `GET /runs`)
- Test: `backend/app/training/tests/test_history_endpoint.py`

**Context:** History = the union of (a) `training_runs` rows (all attempts, incl. failed/in-progress), enriched with recorder metrics where `recorder_id` matches; and (b) recorders that have NO run row (models trained before P2 / by other paths) shown as `status="historical"`. Reuse `evaluation.service.list_recorders_with_summary()` for metrics.

- [ ] **Step 1: Write the failing test**

```python
# backend/app/training/tests/test_history_endpoint.py
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.core.db import get_session


@pytest.mark.asyncio
async def test_runs_endpoint_merges_runs_and_recorders(monkeypatch):
    from app.training import service as svc

    # Fake training_runs rows
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
    app.dependency_overrides[get_session] = lambda: None  # build_history gets session but fake_list_runs ignores it
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/training/runs")
    assert r.status_code == 200
    rows = r.json()
    by_id = {row.get("recorder_id") or row["job_id"]: row for row in rows}
    # j1 (done) enriched with recA metrics:
    assert by_id["recA"]["status"] == "done"
    assert by_id["recA"]["ic_mean"] == 0.04
    # j2 failed, no recorder:
    assert by_id["j2"]["status"] == "failed" and by_id["j2"]["recorder_id"] is None
    # recOld has no run → historical:
    assert by_id["recOld"]["status"] == "historical"
```

> Implementation note for the test to hold: `build_history` must call a module-level `_list_runs(session)` (thin wrapper around `store.list_runs`, monkeypatchable) and the module-level `list_recorders_with_summary` imported into `training/service.py`. Wire those names accordingly in Step 3.

- [ ] **Step 2: Run to verify it fails**

Run (cwd `backend`): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/tests/test_history_endpoint.py -v`
Expected: FAIL — 404 (`/runs` not registered).

- [ ] **Step 3: Add schema + build_history + endpoint**

Add to `backend/app/training/schemas.py`:

```python
class TrainingRunRow(BaseModel):
    job_id: str | None = None
    kind: str | None = None
    scope: str | None = None
    status: str                       # pending|running|done|failed|skipped|historical
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str | None = None
    recorder_id: str | None = None
    run_name: str | None = None
    error: str | None = None
    ic_mean: float | None = None
    ir: float | None = None
    acceptance_passed: bool | None = None
```

Add to `backend/app/training/service.py` (top: `from app.evaluation.service import list_recorders_with_summary` and `from app.training import store` and `from app.training.schemas import TrainingRunRow`; plus the wrapper):

```python
async def _list_runs(session):
    from app.training import store
    return await store.list_runs(session)


async def build_history(session) -> list[TrainingRunRow]:
    """Union of training_runs (all attempts) with recorder summaries (metrics).
    Runs are enriched by matching recorder_id; recorders without a run row are
    appended as status='historical'. Sorted newest-first by created_at."""
    runs = await _list_runs(session)
    try:
        recs = list_recorders_with_summary()
    except Exception:
        recs = []
    rec_by_id = {r.recorder_id: r for r in recs}
    rows: list[TrainingRunRow] = []
    linked: set[str] = set()
    for run in runs:
        rec = rec_by_id.get(run.recorder_id) if run.recorder_id else None
        if rec is not None:
            linked.add(run.recorder_id)
        rows.append(TrainingRunRow(
            job_id=run.job_id, kind=run.kind, scope=run.scope, status=run.status,
            started_at=run.started_at, finished_at=run.finished_at,
            created_at=str(run.created_at) if run.created_at is not None else None,
            recorder_id=run.recorder_id, error=run.error,
            run_name=getattr(rec, "run_name", None),
            ic_mean=getattr(rec, "ic_mean", None), ir=getattr(rec, "ir", None),
            acceptance_passed=getattr(rec, "acceptance_passed", None),
        ))
    for rec in recs:
        if rec.recorder_id in linked:
            continue
        rows.append(TrainingRunRow(
            status="historical", recorder_id=rec.recorder_id, run_name=rec.run_name,
            created_at=rec.created_at, ic_mean=rec.ic_mean, ir=rec.ir,
            acceptance_passed=rec.acceptance_passed,
        ))
    rows.sort(key=lambda r: r.created_at or "", reverse=True)
    return rows
```

Add to `backend/app/training/router.py`:

```python
@router.get("/runs", response_model=list[TrainingRunRow])
async def training_runs(session: AsyncSession = Depends(get_session)):
    from app.training.service import build_history
    return await build_history(session)
```

(Update the router imports: add `TrainingRunRow` to the `from app.training.schemas import ...` line.)

- [ ] **Step 4: Run to verify it passes**

Run (cwd `backend`): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/tests/test_history_endpoint.py app/training/ -v`
Expected: PASS (new test + all prior training tests).

- [ ] **Step 5: Commit**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add backend/app/training/schemas.py backend/app/training/service.py backend/app/training/router.py backend/app/training/tests/test_history_endpoint.py
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "feat(training): GET /api/training/runs — history (runs ∪ recorders)"
```

---

## Task 7: Frontend — runs + rollback client/hooks

**Files:**
- Modify: `frontend/src/api/client.ts` (add `training.runs`; `models.rollback` already exists — confirm)
- Modify: `frontend/src/training/hooks.ts` (add `useTrainingRuns`, `useRollback`)

- [ ] **Step 1: Add the client type + method**

In `frontend/src/api/client.ts`, add the row type near the other training types:

```tsx
export interface TrainingRunRow {
  job_id: string | null;
  kind: string | null;
  scope: string | null;
  status: 'pending' | 'running' | 'done' | 'failed' | 'skipped' | 'historical';
  started_at: string | null;
  finished_at: string | null;
  created_at: string | null;
  recorder_id: string | null;
  run_name: string | null;
  error: string | null;
  ic_mean: number | null;
  ir: number | null;
  acceptance_passed: boolean | null;
}
```

Add to the `training` group:

```tsx
    runs: () => request<TrainingRunRow[]>('/api/training/runs'),
```

- [ ] **Step 2: Add the hooks**

In `frontend/src/training/hooks.ts`:

```tsx
export function useTrainingRuns() {
  return useQuery({
    queryKey: ['training', 'runs'],
    queryFn: () => api.training.runs(),
    staleTime: 10_000,
  });
}

export function useRollback() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (target: 'previous_1' | 'previous_2') => api.models.rollback(target),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['training', 'runs'] });
      qc.invalidateQueries({ queryKey: ['evaluation', 'recorders'] });
    },
  });
}
```

(Confirm `api.models.rollback(target)` exists with that signature — verified in client.ts ~line 248. If its signature differs, match it.)

- [ ] **Step 3: Type-check**

Run (cwd `frontend`): `npm run build` → succeeds.

- [ ] **Step 4: Commit**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add frontend/src/api/client.ts frontend/src/training/hooks.ts
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "feat(training): frontend runs client + useTrainingRuns/useRollback hooks"
```

---

## Task 8: Training.tsx — ③历史 section (table + select + rollback)

**Files:**
- Modify: `frontend/src/pages/Training.tsx`
- Test: none (build-verified; visual)

**Context:** Add a 历史模型 section below 进行中. Render `useTrainingRuns()` as a table (reuse the dark Tailwind table style from `portfolio/HoldingsTable.tsx`). Columns: 时间(created_at) · 范围(scope→全量) · 状态 · IC · IR · 验收 · recorder. Multi-select up to 2 (FIFO) via checkboxes for the ④对比 section (lift selection state to the page). A 回滚上一周 button using `useRollback()` + `confirm()` + `toast` (mirror `RetrainScheduleEditor`).

- [ ] **Step 1: Add the history section**

Add to `frontend/src/pages/Training.tsx`. Introduce selection state `const [selected, setSelected] = useState<string[]>([])` and a `toggle(recorderId)` (FIFO max 2, only selectable when `recorder_id` is set). Render:

```tsx
import { useState } from 'react';
import { useActiveTrainingJob, useStartTraining, useTrainingRuns, useRollback } from '@/training/hooks';
import { toast } from '@/jobs/toast';

const STATUS_LABEL: Record<string, string> = {
  pending: '排队', running: '训练中', done: '完成', failed: '失败', skipped: '跳过', historical: '历史',
};

// inside the component, after the 进行中 <section>:
function fmt(x: number | null, d = 3) { return x == null ? '—' : x.toFixed(d); }
```

```tsx
      {/* ③ 历史模型 */}
      <section className="rounded-lg border border-[#30363d] p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-medium text-[#8b949e]">历史模型</h2>
          <button
            className="px-2 py-1 rounded text-xs bg-red-700 text-white hover:bg-red-600 disabled:opacity-50"
            disabled={rollback.isPending}
            onClick={() => {
              if (confirm('回滚到上一版模型？当前 recorder 会被归档。')) {
                const tid = toast.info('正在回滚…', -1);
                rollback.mutate('previous_1', {
                  onSuccess: (r) => { toast.success(`已回滚:${r.status}`); },
                  onError: (e) => { toast.error(`回滚失败:${String((e as Error)?.message ?? e)}`); },
                  onSettled: () => { /* toast auto-replaces */ void tid; },
                });
              }
            }}
          >
            {rollback.isPending ? '回滚中…' : '回滚上一版'}
          </button>
        </div>
        {runs.isLoading && <p className="text-xs text-[#8b949e]">加载中…</p>}
        {runs.data && runs.data.length === 0 && <p className="text-xs text-[#8b949e]">暂无历史。</p>}
        {runs.data && runs.data.length > 0 && (
          <div className="rounded-lg border border-[#30363d] overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-[#161b22] text-[#8b949e] text-xs">
                <tr>
                  <th className="p-2 text-left w-8"></th>
                  <th className="p-2 text-left">时间</th>
                  <th className="p-2 text-left">范围</th>
                  <th className="p-2 text-left">状态</th>
                  <th className="p-2 text-right">IC</th>
                  <th className="p-2 text-right">IR</th>
                  <th className="p-2 text-center">验收</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.map((row) => {
                  const key = row.recorder_id ?? row.job_id ?? Math.random().toString();
                  const selectable = !!row.recorder_id;
                  return (
                    <tr key={key} className="border-t border-[#21262d] hover:bg-[#161b22]">
                      <td className="p-2">
                        <input
                          type="checkbox"
                          disabled={!selectable}
                          checked={!!row.recorder_id && selected.includes(row.recorder_id)}
                          onChange={() => row.recorder_id && toggle(row.recorder_id)}
                        />
                      </td>
                      <td className="p-2 text-[#8b949e]">{(row.created_at ?? '').slice(0, 16).replace('T', ' ')}</td>
                      <td className="p-2">{row.scope === 'full' ? '全量' : row.scope ?? '—'}</td>
                      <td className="p-2">{STATUS_LABEL[row.status] ?? row.status}{row.status === 'failed' && row.error ? ` · ${row.error.slice(0, 40)}` : ''}</td>
                      <td className="p-2 text-right font-mono">{fmt(row.ic_mean)}</td>
                      <td className="p-2 text-right font-mono">{fmt(row.ir, 2)}</td>
                      <td className="p-2 text-center">{row.acceptance_passed == null ? '—' : row.acceptance_passed ? '✓' : '✗'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
```

Wire the hooks at the top of the component: `const runs = useTrainingRuns(); const rollback = useRollback();` and the selection helper:

```tsx
  const [selected, setSelected] = useState<string[]>([]);
  const toggle = (rid: string) =>
    setSelected((prev) =>
      prev.includes(rid) ? prev.filter((x) => x !== rid) : prev.length >= 2 ? [prev[1], rid] : [...prev, rid],
    );
```

- [ ] **Step 2: Build**

Run (cwd `frontend`): `npm run build` → succeeds.

- [ ] **Step 3: Commit**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add frontend/src/pages/Training.tsx
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "feat(training): 训练工作台 历史模型 table (select + rollback)"
```

---

## Task 9: Training.tsx — ④对比 section (reuse evaluation compare)

**Files:**
- Modify: `frontend/src/pages/Training.tsx`
- (Reuse: `frontend/src/pages/evaluation/hooks.ts::useCompare`, `frontend/src/pages/evaluation/CompareCard.tsx`)

**Context:** When exactly 2 recorders are selected in ③, render a 对比 section using the existing `useCompare(a, b)` hook + `CompareCard` component (it already renders the paired-t verdict + per-metric deltas). This is pure reuse — do not rebuild compare.

- [ ] **Step 1: Add the compare section**

```tsx
import { useCompare } from '@/pages/evaluation/hooks';
import CompareCard from '@/pages/evaluation/CompareCard';
```

```tsx
      {/* ④ 对比 */}
      {selected.length === 2 && <CompareSection a={selected[0]} b={selected[1]} />}
```

Add a small wrapper component (in the same file, below `Training`):

```tsx
function CompareSection({ a, b }: { a: string; b: string }) {
  const cmp = useCompare(a, b);
  return (
    <section className="rounded-lg border border-[#30363d] p-4 space-y-3">
      <h2 className="text-sm font-medium text-[#8b949e]">对比(已选 2 个)</h2>
      {cmp.isLoading && <p className="text-xs text-[#8b949e]">评估/对比计算中…(首次较慢)</p>}
      {cmp.isError && <p className="text-xs text-red-400">对比失败:{String((cmp.error as Error)?.message ?? cmp.error)}</p>}
      {cmp.data && <CompareCard data={cmp.data} />}
    </section>
  );
}
```

> Confirm `useCompare`'s return + `CompareCard`'s prop name by reading `frontend/src/pages/evaluation/hooks.ts` and `CompareCard.tsx`. If `CompareCard` takes props other than `data` (e.g. `result`), match the real signature. If `useCompare` needs options (top_k/cost_bps), pass the same defaults the Evaluation page uses.

- [ ] **Step 2: Build**

Run (cwd `frontend`): `npm run build` → succeeds.

- [ ] **Step 3: Commit**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add frontend/src/pages/Training.tsx
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "feat(training): 训练工作台 对比 section (reuse useCompare + CompareCard)"
```

---

## Task 10: Regression + build + live smoke

**Files:** none (verification).

- [ ] **Step 1: Backend — training + scheduling suites**

Run (cwd `backend`): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/ app/scheduling/ -v`
Expected: all pass (P1 + all P2 backend tests). (The data-dependent suites — charts/data/evaluation/models/core — can only pass in a worktree with qlib data; not a P2 regression.)

- [ ] **Step 2: Frontend build**

Run (cwd `frontend`): `npm run build` → clean tsc + vite build.

- [ ] **Step 3: Live smoke (data-bearing env)**

In a checkout WITH qlib data (e.g. main after merge), restart the backend, open 训练工作台:
- 历史模型 table lists recorders with IC/IR/验收; trigger a real 立即训练(全量) → after it finishes a new row appears with status 完成 + a recorder id.
- Select 2 recorders → 对比 section shows the CompareCard verdict + metric deltas.
- 回滚上一版 → confirm → toast success; the current model reverts (verify via the version badge / next screen).

- [ ] **Step 4: Commit any smoke fixes**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add -A
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "fix(training): P2 live-smoke findings"
```

---

## Self-Review (completed during authoring)
- **Spec coverage:** history (T6 endpoint + T8 table), evaluation (reused), compare (T9 reuse), rollback (T7/T8 reuse), training_runs persistence (T1-T2-T5), recorder linkage (T3 emit + T4 parse + T5 store). ✓
- **Promote/candidate + loss curves:** explicitly deferred to P3 (stated in header). ✓
- **Placeholder scan:** every code step has complete code; the two "confirm the real signature" notes (T7 rollback, T9 useCompare/CompareCard) are verification instructions with a concrete fallback, not placeholders. ✓
- **Type consistency:** `TrainingRunRow` identical across schemas.py / build_history / client.ts / Training.tsx; `record_run_start/finish/list_runs` signatures match across store.py, test, and the SchedulerManager `_persist_run` caller; `latest_recorder_id` ↔ `emit_recorder` (`RECORDER ` prefix) consistent. ✓
- **Import-cycle safety:** scheduling→training imports are all LAZY (inside `_persist_run`); training→scheduling stays at module level (router get_manager). ✓
- **Risk:** T5 (SchedulerManager↔DB) is the highest-risk task — the test drives `_gated_job_fn` directly with a monkeypatched `_session_maker` to verify a row is written end-to-end.
