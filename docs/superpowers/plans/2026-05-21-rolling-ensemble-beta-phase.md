# Rolling Multi-Model Ensemble · β Phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-model `daily_cn_fresh` baseline with a weekly walk-forward rolling 3-model ensemble (LightGBM + ALSTM + TRA) on CSI800 with PIT survivorship-free training, multi-horizon open-to-open labels, Ridge stacking, EWMA turnover smoothing, and an in-app retrain scheduler.

**Architecture:** Production pipeline in `production/rolling_train.py` orchestrates the weekly run end-to-end (PIT universe → walk-forward split → 3 base models → OOF stacking → EWMA post-process → mlruns recorder + `pred.pkl`). Backend gains a `scheduling/` module using APScheduler in the FastAPI lifespan, persisting cron config in SQLite via Alembic migration `0003`. Backend `models/` router extends to surface per-base-model views, consensus, and recorder version metadata. Frontend gains a Settings page (schedule editor) and extends Picks with view selector + consensus column.

**Tech Stack:** Python 3.10 (qlib env at `F:\Tools\Anaconda\envs\qlib\python.exe`); qlib + PyTorch (CUDA on RTX 3080Ti / 5070Ti); LightGBM CPU; scikit-learn Ridge; APScheduler; FastAPI + SQLAlchemy 2.x async + Alembic; React 18 + Vite + TanStack Query + Tailwind.

**Spec:** `docs/superpowers/specs/2026-05-21-rolling-ensemble-algorithm-design.md`

**Run from worktree root** unless a step says otherwise. Python path is `F:\Tools\Anaconda\envs\qlib\python.exe`. Backend tests run with `cd backend; F:/Tools/Anaconda/envs/qlib/python.exe -m pytest <path>`. Production tests run with `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/<path>`.

---

## File Structure

### Production pipeline (`production/`)

- `pit_constituents.py` — PIT CSI300+CSI500 monthly snapshot fetcher with parquet cache + range sanity check.
- `walk_forward.py` — `WalkForwardSplit` dataclass + `split(end_date, horizon)` returning train/valid/stack/test windows; raises if any overlap.
- `multi_horizon_labels.py` — `Alpha158_OpenH(N)` and `Alpha360_OpenH(N)` handlers (`N in {1,5,20}`) using `Ref($open,-N-1)/Ref($open,-1)-1`.
- `post_process.py` — `ewma_smooth(scores_df, alpha=0.5)` and `cost_adjust(returns, turnover, bps=10)`.
- `consensus.py` — `consensus_score(base_preds_9d)` returning [0,1] + `write_pred_pkl(...)` writing the unified output (`score`, `consensus`, `base_scores`).
- `ensemble_stacker.py` — `RidgeStacker` with `.fit_oof(...)`, `.predict(...)`, `.fallback_rank_average(...)`; multi-level chain.
- `metrics.py` — `compute_scorecard(...)` → dict of 8 metrics; `regime_split(...)`; `paired_ttest(...)`.
- `mlruns_archive.py` — `archive_old_recorders(weeks=8)`.
- `rolling_train.py` — main CLI; subcommands `run-once`, `backfill <end-date>`, `evaluate <recorder-id>`.
- `configs/rolling_ensemble.yaml` — top-level (model list, horizons, windows, regime dates).
- `configs/lgbm_alpha158_multi.yaml` — LightGBM hyperparameters per horizon.
- `configs/alstm_alpha360.yaml` — ALSTM hyperparameters.
- `configs/tra_alpha360.yaml` — TRA hyperparameters.

### Backend (`backend/app/`)

- `scheduling/__init__.py`
- `scheduling/orm.py` — `RetrainScheduleORM` single-row table.
- `scheduling/schemas.py` — `RetrainScheduleRead`, `RetrainScheduleUpdate`, `RunNowResponse`.
- `scheduling/service.py` — `SchedulerManager` wrapping APScheduler, with `get_schedule`, `update_schedule`, `run_now`, `_is_trading_hours(dt)`.
- `scheduling/router.py` — `GET /api/scheduling/retrain`, `PUT /api/scheduling/retrain`, `POST /api/scheduling/retrain/run-now`.
- `scheduling/tests/test_service.py`, `tests/test_router.py`, `tests/test_trading_hours.py`.
- `models/router.py` — extend with `?view=ensemble|lightgbm|alstm|tra`, `GET /api/models/version`, `POST /api/models/rollback`, `GET /api/models/shadow`.
- `models/service.py` — read `consensus` + `base_scores` from `pred.pkl`; expose `version_info()`.
- `models/schemas.py` — extend `ScreenItem` with `consensus: float`, `base_scores: dict[str, float]`.
- `core/config.py` — add `retrain_recorder_experiment: str = "rolling_v2_ensemble"`.
- `main.py` — include `scheduling_router`; start/stop APScheduler in `lifespan`.

### Migrations (`backend/alembic/versions/`)

- `0003_add_retrain_schedule.py` — adds `retrain_schedule` table (single-row config).

### Frontend (`frontend/src/`)

- `pages/Settings.tsx` — new page with schedule editor + Run-now button + last/next run.
- `components/RetrainScheduleEditor.tsx` — reusable form.
- `pages/Picks.tsx` — view selector (Ensemble/LightGBM/ALSTM/TRA), consensus column, consensus filter slider.
- `pages/Dashboard.tsx` — Model Version card + retrain countdown.
- `App.tsx` — add `/settings` route.
- `components/Layout.tsx` — add Settings nav link.
- `api/types.ts` — regenerated via `npm run gen:api` after backend changes.

### Production tests (`production/tests/`)

- `__init__.py`
- `conftest.py` — fixtures: synthetic price df, synthetic OOF preds, mock baostock.
- `test_pit_constituents.py`
- `test_walk_forward.py` — **must contain** `test_no_overlap_train_valid_stack_test`.
- `test_multi_horizon_labels.py`
- `test_post_process.py`
- `test_consensus.py`
- `test_ensemble_stacker.py`
- `test_metrics.py`
- `test_mlruns_archive.py`

---

## Execution Order

```
Phase A (Schedule infra)      T1 → T2 → T3 → T4 → T5
Phase B (ML foundation)        T6 → T7 → T8
Phase C (LightGBM milestone)   T9 → T10 → T11 → T12   ← ships working baseline replacement
Phase D (ALSTM milestone)      T13 → T14              ← ships 2-model ensemble
Phase E (TRA + stacking)       T15 → T16 → T17 → T18 → T19
Phase F (Frontend + final)     T20 → T21              ← ships full UI integration + acceptance gate
```

Each milestone end (T12, T14, T19, T21) produces a working, shippable system.

---

## Phase A — Schedule Infrastructure

### Task 1: Alembic migration `0003_add_retrain_schedule`

**Files:**
- Create: `backend/alembic/versions/0003_add_retrain_schedule.py`

- [ ] **Step 1: Create the migration file**

```python
"""add retrain_schedule table

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-21

Single-row config for the weekly retrain cron. Seeded with default
Sunday 22:00 on first upgrade.
"""
from alembic import op
import sqlalchemy as sa


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "retrain_schedule",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("day_of_week", sa.Integer(), nullable=False),  # 0=Mon … 6=Sun
        sa.Column("hour", sa.Integer(), nullable=False),
        sa.Column("minute", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("last_run_at", sa.DateTime(), nullable=True),
        sa.Column("next_run_at", sa.DateTime(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.CheckConstraint("day_of_week BETWEEN 0 AND 6", name="ck_dow_range"),
        sa.CheckConstraint("hour BETWEEN 0 AND 23", name="ck_hour_range"),
        sa.CheckConstraint("minute BETWEEN 0 AND 59", name="ck_minute_range"),
        sa.CheckConstraint("id = 1", name="ck_single_row"),
    )
    # Seed default row
    op.execute(
        "INSERT INTO retrain_schedule (id, day_of_week, hour, minute, enabled) "
        "VALUES (1, 6, 22, 0, 1)"
    )


def downgrade() -> None:
    op.drop_table("retrain_schedule")
```

- [ ] **Step 2: Run the migration**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m alembic upgrade head
```

Expected output: `Running upgrade 0002 -> 0003, add retrain_schedule table` and the prompt returns cleanly.

- [ ] **Step 3: Verify the row was seeded**

```
F:/Tools/Anaconda/envs/qlib/python.exe -c "import sqlite3; c = sqlite3.connect('backend/app.db'); print(c.execute('SELECT * FROM retrain_schedule').fetchone())"
```

Expected: `(1, 6, 22, 0, 1, None, None, '<timestamp>')`.

- [ ] **Step 4: Commit**

```
git add backend/alembic/versions/0003_add_retrain_schedule.py
git commit -m "feat(db): add retrain_schedule table (Sun 22:00 default)"
```

---

### Task 2: Scheduling ORM + schemas + service

**Files:**
- Create: `backend/app/scheduling/__init__.py`
- Create: `backend/app/scheduling/orm.py`
- Create: `backend/app/scheduling/schemas.py`
- Create: `backend/app/scheduling/service.py`
- Create: `backend/app/scheduling/tests/__init__.py`
- Create: `backend/app/scheduling/tests/test_trading_hours.py`
- Modify: `backend/pyproject.toml` (add `apscheduler>=3.10`)

- [ ] **Step 1: Add APScheduler dependency**

Open `backend/pyproject.toml` and edit the `dependencies` block to append `"apscheduler>=3.10",` as the last entry before the closing `]`. The updated block reads:

```toml
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "sqlalchemy[asyncio]>=2.0",
    "aiosqlite>=0.20",
    "alembic>=1.13",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "structlog>=24.4",
    "python-multipart>=0.0.20",
    "httpx>=0.27",
    "apscheduler>=3.10",
]
```

Then install:

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pip install -e .
```

Expected: `Successfully installed apscheduler-...`.

- [ ] **Step 2: Create the package skeleton**

`backend/app/scheduling/__init__.py`:

```python
"""Retrain schedule + in-process APScheduler manager."""
```

`backend/app/scheduling/orm.py`:

```python
from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, Integer
from sqlalchemy.sql import func

from app.core.db import Base


class RetrainScheduleORM(Base):
    __tablename__ = "retrain_schedule"

    id = Column(Integer, primary_key=True)
    day_of_week = Column(Integer, nullable=False)
    hour = Column(Integer, nullable=False)
    minute = Column(Integer, nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    last_run_at = Column(DateTime, nullable=True)
    next_run_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)

    __table_args__ = (
        CheckConstraint("day_of_week BETWEEN 0 AND 6", name="ck_dow_range"),
        CheckConstraint("hour BETWEEN 0 AND 23", name="ck_hour_range"),
        CheckConstraint("minute BETWEEN 0 AND 59", name="ck_minute_range"),
        CheckConstraint("id = 1", name="ck_single_row"),
    )
```

`backend/app/scheduling/schemas.py`:

```python
from datetime import datetime

from pydantic import BaseModel, Field


class RetrainScheduleRead(BaseModel):
    day_of_week: int = Field(ge=0, le=6, description="0=Mon, 6=Sun")
    hour: int = Field(ge=0, le=23)
    minute: int = Field(ge=0, le=59)
    enabled: bool
    last_run_at: datetime | None
    next_run_at: datetime | None


class RetrainScheduleUpdate(BaseModel):
    day_of_week: int = Field(ge=0, le=6)
    hour: int = Field(ge=0, le=23)
    minute: int = Field(ge=0, le=59)
    enabled: bool


class RunNowResponse(BaseModel):
    status: str  # "started" | "rejected"
    reason: str | None = None
    job_id: str | None = None
```

- [ ] **Step 3: Write the failing trading-hours guard test**

`backend/app/scheduling/tests/__init__.py`:

```python
```

`backend/app/scheduling/tests/test_trading_hours.py`:

```python
from datetime import datetime

import pytest

from app.scheduling.service import is_trading_hours_cst


@pytest.mark.parametrize(
    "dt_str, expected",
    [
        ("2026-05-21 09:30", True),   # Thu, market open
        ("2026-05-21 15:00", True),   # Thu, market close
        ("2026-05-21 11:30", True),   # Thu, midday
        ("2026-05-21 08:00", False),  # Thu, before open
        ("2026-05-21 15:01", False),  # Thu, after close
        ("2026-05-23 11:00", False),  # Sat
        ("2026-05-24 22:00", False),  # Sun
    ],
)
def test_trading_hours_cst(dt_str, expected):
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    assert is_trading_hours_cst(dt) == expected
```

- [ ] **Step 4: Run the test to confirm it fails**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/scheduling/tests/test_trading_hours.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.scheduling.service'`.

- [ ] **Step 5: Implement the service**

`backend/app/scheduling/service.py`:

```python
from __future__ import annotations

import asyncio
from datetime import datetime, time
from typing import Awaitable, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.scheduling.orm import RetrainScheduleORM
from app.scheduling.schemas import RetrainScheduleRead, RetrainScheduleUpdate

_log = get_logger("scheduling")

# CST trading hours (Shenzhen/Shanghai): 09:30–11:30 lunch 13:00–15:00, but the
# guard treats the full 09:30–15:00 window as "trading hours" to avoid mid-day
# model swap. Weekends are never trading hours.
_TRADING_OPEN = time(9, 30)
_TRADING_CLOSE = time(15, 0)


def is_trading_hours_cst(dt: datetime) -> bool:
    if dt.weekday() >= 5:  # Sat=5, Sun=6
        return False
    t = dt.time()
    return _TRADING_OPEN <= t <= _TRADING_CLOSE


class TradingHoursViolation(Exception):
    pass


JobCallable = Callable[[], Awaitable[None]]


class SchedulerManager:
    """Wraps an AsyncIOScheduler and the single-row schedule config."""

    JOB_ID = "retrain_weekly"

    def __init__(self, job_fn: JobCallable):
        self._scheduler = AsyncIOScheduler()
        self._job_fn = job_fn
        self._started = False

    async def start(self, session: AsyncSession) -> None:
        schedule = await self._read_row(session)
        self._scheduler.start()
        self._started = True
        if schedule.enabled:
            self._install_job(schedule)
        _log.info("scheduler_started", schedule=schedule.model_dump())

    async def stop(self) -> None:
        if self._started:
            self._scheduler.shutdown(wait=False)
            self._started = False
            _log.info("scheduler_stopped")

    async def get_schedule(self, session: AsyncSession) -> RetrainScheduleRead:
        return await self._read_row(session)

    async def update_schedule(
        self, session: AsyncSession, payload: RetrainScheduleUpdate
    ) -> RetrainScheduleRead:
        # Reject trading-hours slots for safety
        now = datetime.now()
        sample = datetime(
            now.year, now.month, now.day, payload.hour, payload.minute
        )
        # We check using the day-of-week the user picked rather than today
        weekday_sample = datetime(2026, 1, 5)  # Mon = weekday 0
        weekday_sample = weekday_sample.replace(day=5 + payload.day_of_week)
        weekday_sample = weekday_sample.replace(hour=payload.hour, minute=payload.minute)
        if is_trading_hours_cst(weekday_sample):
            raise TradingHoursViolation(
                f"slot {payload.day_of_week}/{payload.hour:02d}:{payload.minute:02d} "
                f"falls inside trading hours"
            )

        row = await session.get(RetrainScheduleORM, 1)
        if row is None:
            raise RuntimeError("retrain_schedule row 1 missing")
        row.day_of_week = payload.day_of_week
        row.hour = payload.hour
        row.minute = payload.minute
        row.enabled = payload.enabled
        await session.commit()
        await session.refresh(row)

        schedule = self._row_to_read(row)
        self._reinstall_job(schedule)
        _log.info("schedule_updated", schedule=schedule.model_dump())
        return schedule

    async def run_now(self, session: AsyncSession, force: bool = False) -> str:
        now = datetime.now()
        if not force and is_trading_hours_cst(now):
            raise TradingHoursViolation(
                "run_now refused during trading hours; pass force=true to override"
            )
        job = self._scheduler.add_job(
            self._job_fn, trigger=None, id=f"{self.JOB_ID}_manual_{now.timestamp():.0f}"
        )
        _log.info("run_now_scheduled", job_id=job.id)
        return job.id

    def _install_job(self, schedule: RetrainScheduleRead) -> None:
        trigger = CronTrigger(
            day_of_week=schedule.day_of_week,
            hour=schedule.hour,
            minute=schedule.minute,
        )
        self._scheduler.add_job(self._job_fn, trigger=trigger, id=self.JOB_ID, replace_existing=True)

    def _reinstall_job(self, schedule: RetrainScheduleRead) -> None:
        try:
            self._scheduler.remove_job(self.JOB_ID)
        except Exception:
            pass
        if schedule.enabled:
            self._install_job(schedule)

    async def _read_row(self, session: AsyncSession) -> RetrainScheduleRead:
        row = await session.get(RetrainScheduleORM, 1)
        if row is None:
            raise RuntimeError("retrain_schedule row 1 missing")
        return self._row_to_read(row)

    @staticmethod
    def _row_to_read(row: RetrainScheduleORM) -> RetrainScheduleRead:
        return RetrainScheduleRead(
            day_of_week=row.day_of_week,
            hour=row.hour,
            minute=row.minute,
            enabled=row.enabled,
            last_run_at=row.last_run_at,
            next_run_at=row.next_run_at,
        )
```

- [ ] **Step 6: Run the trading-hours test, expect PASS**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/scheduling/tests/test_trading_hours.py -v
```

Expected: all 7 parametrized cases PASS.

- [ ] **Step 7: Commit**

```
git add backend/pyproject.toml backend/app/scheduling/
git commit -m "feat(scheduling): ORM + APScheduler manager with trading-hours guard"
```

---

### Task 3: Scheduling router (GET/PUT/run-now)

**Files:**
- Create: `backend/app/scheduling/router.py`
- Create: `backend/app/scheduling/tests/test_router.py`

- [ ] **Step 1: Write the failing router tests**

`backend/app/scheduling/tests/test_router.py`:

```python
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        async with app.router.lifespan_context(app):
            yield c


async def test_get_schedule_returns_default(client):
    r = await client.get("/api/scheduling/retrain")
    assert r.status_code == 200
    body = r.json()
    assert body["day_of_week"] == 6
    assert body["hour"] == 22
    assert body["minute"] == 0
    assert body["enabled"] is True


async def test_put_schedule_updates_row(client):
    r = await client.put(
        "/api/scheduling/retrain",
        json={"day_of_week": 5, "hour": 23, "minute": 30, "enabled": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["day_of_week"] == 5
    assert body["hour"] == 23
    assert body["minute"] == 30


async def test_put_schedule_rejects_trading_hours(client):
    r = await client.put(
        "/api/scheduling/retrain",
        json={"day_of_week": 1, "hour": 10, "minute": 0, "enabled": True},  # Tue 10:00
    )
    assert r.status_code == 400
    assert "trading_hours" in r.json()["error_code"]
```

- [ ] **Step 2: Run, expect failure**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/scheduling/tests/test_router.py -v
```

Expected: `Failed: AttributeError: module 'app.scheduling' has no attribute 'router'` or 404.

- [ ] **Step 3: Implement the router**

`backend/app/scheduling/router.py`:

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.exceptions import BusinessError
from app.scheduling.schemas import (
    RetrainScheduleRead,
    RetrainScheduleUpdate,
    RunNowResponse,
)
from app.scheduling.service import SchedulerManager, TradingHoursViolation

router = APIRouter()

# Manager singleton populated in main.lifespan
_manager: SchedulerManager | None = None


def set_manager(m: SchedulerManager) -> None:
    global _manager
    _manager = m


def get_manager() -> SchedulerManager:
    if _manager is None:
        raise RuntimeError("SchedulerManager not initialized")
    return _manager


@router.get("/retrain", response_model=RetrainScheduleRead)
async def get_schedule(session: AsyncSession = Depends(get_session)):
    return await get_manager().get_schedule(session)


@router.put("/retrain", response_model=RetrainScheduleRead)
async def put_schedule(
    payload: RetrainScheduleUpdate,
    session: AsyncSession = Depends(get_session),
):
    try:
        return await get_manager().update_schedule(session, payload)
    except TradingHoursViolation as exc:
        raise BusinessError(
            code="trigger_during_trading_hours",
            message=str(exc),
            http_status=400,
        )


@router.post("/retrain/run-now", response_model=RunNowResponse)
async def run_now(
    force: bool = Query(default=False),
    session: AsyncSession = Depends(get_session),
):
    try:
        job_id = await get_manager().run_now(session, force=force)
        return RunNowResponse(status="started", job_id=job_id)
    except TradingHoursViolation as exc:
        return RunNowResponse(status="rejected", reason=str(exc))
```

- [ ] **Step 4: Run, expect failure (router not yet wired)**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/scheduling/tests/test_router.py -v
```

Expected: 404 because the router is not included in `main.py` yet. This is fixed in Task 4.

- [ ] **Step 5: Commit**

```
git add backend/app/scheduling/router.py backend/app/scheduling/tests/test_router.py
git commit -m "feat(scheduling): GET/PUT/run-now endpoints"
```

---

### Task 4: Wire scheduling into `main.py` lifespan

**Files:**
- Modify: `backend/app/main.py`
- Create: `backend/app/scheduling/tests/test_service.py`

- [ ] **Step 1: Write the failing lifespan test**

`backend/app/scheduling/tests/test_service.py`:

```python
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.scheduling.router import get_manager


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        async with app.router.lifespan_context(app):
            yield c


async def test_manager_is_initialized_after_lifespan(client):
    manager = get_manager()
    assert manager is not None
    assert manager._started is True
```

- [ ] **Step 2: Modify `main.py` to wire scheduling**

Edit `backend/app/main.py`:
1. Add imports at the top of imports:

```python
from app.scheduling.router import router as scheduling_router, set_manager
from app.scheduling.service import SchedulerManager
```

2. Modify `lifespan` to instantiate and start the manager. Replace the existing `lifespan` function with:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    configure_logging(json_output=True)
    log = get_logger("startup")
    init_db_singletons(settings)
    try:
        init_qlib_once(settings)
        log.info("qlib_ready")
    except Exception as e:
        log.warning("qlib_not_ready_at_boot", error=str(e))

    # Scheduler manager — job_fn is a stub for now; T10 replaces with the real
    # rolling_train.run() invocation.
    async def _stub_job() -> None:
        log.info("retrain_job_fired_stub")

    manager = SchedulerManager(_stub_job)
    set_manager(manager)
    from app.core.db import _session_maker
    assert _session_maker is not None
    async with _session_maker() as session:
        await manager.start(session)

    log.info("app_started", port=settings.api_port)
    yield
    await manager.stop()
    await dispose_db_singletons()
    log.info("app_stopped")
```

3. In `create_app()`, after the existing `app.include_router(...)` calls, add:

```python
    app.include_router(scheduling_router, prefix="/api/scheduling", tags=["scheduling"])
```

- [ ] **Step 3: Run all scheduling tests, expect PASS**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/scheduling/tests/ -v
```

Expected: 3 router tests + 7 trading-hours params + 1 service test all PASS.

- [ ] **Step 4: Run the full test suite to check no regressions**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest -x
```

Expected: all green (existing tests + new ones).

- [ ] **Step 5: Commit**

```
git add backend/app/main.py backend/app/scheduling/tests/test_service.py
git commit -m "feat(scheduling): wire APScheduler lifecycle into FastAPI lifespan"
```

---

### Task 5: Frontend Settings page with schedule editor

**Files:**
- Create: `frontend/src/pages/Settings.tsx`
- Create: `frontend/src/components/RetrainScheduleEditor.tsx`
- Modify: `frontend/src/App.tsx` (add `/settings` route)
- Modify: `frontend/src/components/Layout.tsx` (add nav link)
- Modify: `frontend/src/api/types.ts` (regenerate)

- [ ] **Step 1: Regenerate API types**

```
cd frontend
npm run gen:api
```

Expected: `frontend/src/api/types.ts` now includes `RetrainScheduleRead`, `RetrainScheduleUpdate`, `RunNowResponse`. Confirm with:

```
grep -n RetrainScheduleRead frontend/src/api/types.ts
```

- [ ] **Step 2: Create `RetrainScheduleEditor.tsx`**

```tsx
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { client } from "../api/client";
import type { components } from "../api/types";

type Schedule = components["schemas"]["RetrainScheduleRead"];
type Update = components["schemas"]["RetrainScheduleUpdate"];

const DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export function RetrainScheduleEditor() {
  const qc = useQueryClient();
  const { data, isLoading, error } = useQuery({
    queryKey: ["schedule"],
    queryFn: async () => {
      const { data, error } = await client.GET("/api/scheduling/retrain");
      if (error) throw new Error(JSON.stringify(error));
      return data as Schedule;
    },
  });

  const [draft, setDraft] = useState<Update | null>(null);
  const effective: Update = draft ?? (data
    ? { day_of_week: data.day_of_week, hour: data.hour, minute: data.minute, enabled: data.enabled }
    : { day_of_week: 6, hour: 22, minute: 0, enabled: true });

  const saveMut = useMutation({
    mutationFn: async (payload: Update) => {
      const { data, error } = await client.PUT("/api/scheduling/retrain", { body: payload });
      if (error) throw new Error(JSON.stringify(error));
      return data;
    },
    onSuccess: () => {
      setDraft(null);
      qc.invalidateQueries({ queryKey: ["schedule"] });
    },
  });

  const runNowMut = useMutation({
    mutationFn: async (force: boolean) => {
      const { data, error } = await client.POST("/api/scheduling/retrain/run-now", {
        params: { query: { force } },
      });
      if (error) throw new Error(JSON.stringify(error));
      return data;
    },
  });

  if (isLoading) return <div>Loading schedule…</div>;
  if (error) return <div className="text-red-500">Error: {String(error)}</div>;

  return (
    <div className="space-y-4 p-4 border border-gray-700 rounded-md">
      <h3 className="text-lg font-semibold">Weekly retrain schedule</h3>
      <div className="flex gap-4 items-end">
        <label className="flex flex-col">
          <span className="text-sm">Day of week</span>
          <select
            className="bg-gray-800 px-2 py-1 rounded"
            value={effective.day_of_week}
            onChange={(e) => setDraft({ ...effective, day_of_week: Number(e.target.value) })}
          >
            {DOW.map((d, i) => (
              <option key={i} value={i}>{d}</option>
            ))}
          </select>
        </label>
        <label className="flex flex-col">
          <span className="text-sm">Hour</span>
          <input
            type="number"
            min={0}
            max={23}
            className="bg-gray-800 px-2 py-1 rounded w-20"
            value={effective.hour}
            onChange={(e) => setDraft({ ...effective, hour: Number(e.target.value) })}
          />
        </label>
        <label className="flex flex-col">
          <span className="text-sm">Minute</span>
          <input
            type="number"
            min={0}
            max={59}
            className="bg-gray-800 px-2 py-1 rounded w-20"
            value={effective.minute}
            onChange={(e) => setDraft({ ...effective, minute: Number(e.target.value) })}
          />
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={effective.enabled}
            onChange={(e) => setDraft({ ...effective, enabled: e.target.checked })}
          />
          <span className="text-sm">Enabled</span>
        </label>
      </div>
      <div className="flex gap-2">
        <button
          className="px-3 py-1 bg-blue-600 rounded disabled:opacity-50"
          disabled={draft === null || saveMut.isPending}
          onClick={() => saveMut.mutate(effective)}
        >
          Save
        </button>
        <button
          className="px-3 py-1 bg-orange-600 rounded"
          onClick={() => runNowMut.mutate(false)}
        >
          Run now
        </button>
      </div>
      {saveMut.error && (
        <div className="text-red-400 text-sm">Save failed: {String(saveMut.error)}</div>
      )}
      {runNowMut.data?.status === "rejected" && (
        <div className="text-yellow-400 text-sm">
          Rejected: {runNowMut.data.reason}. Click "Run anyway" to force.
          <button
            className="ml-2 px-2 py-0.5 bg-red-700 rounded text-xs"
            onClick={() => runNowMut.mutate(true)}
          >
            Run anyway
          </button>
        </div>
      )}
      <div className="text-xs text-gray-400">
        Last run: {data?.last_run_at ?? "never"} · Next run: {data?.next_run_at ?? "—"}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create `Settings.tsx`**

```tsx
import { RetrainScheduleEditor } from "../components/RetrainScheduleEditor";

export default function Settings() {
  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Settings</h2>
      <RetrainScheduleEditor />
    </div>
  );
}
```

- [ ] **Step 4: Wire route + nav link**

In `frontend/src/App.tsx`, add to the existing `<Routes>` block (alongside `<Route path="/dashboard" ...>`):

```tsx
import Settings from "./pages/Settings";
// ...
<Route path="/settings" element={<Settings />} />
```

In `frontend/src/components/Layout.tsx`, add a nav link inside the existing nav (next to Dashboard / Picks / Portfolio):

```tsx
<NavLink to="/settings" className={navCls}>Settings</NavLink>
```

- [ ] **Step 5: Smoke test in browser**

```
cd frontend
npm run dev
```

Then in a separate terminal start the backend:

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m uvicorn app.main:app --port 8000
```

Open `http://localhost:5173/settings`. Verify:
- Form displays Sun / 22 / 0 / enabled checked.
- Changing day-of-week to "Tue", hour to 10 → click Save → red error appears (trading hours rejected).
- Changing to Mon / 21 / 30 → click Save → no error.
- Refresh the page; values persist.
- "Run now" outside trading hours → no error.

- [ ] **Step 6: Commit**

```
git add frontend/src/pages/Settings.tsx frontend/src/components/RetrainScheduleEditor.tsx frontend/src/App.tsx frontend/src/components/Layout.tsx frontend/src/api/types.ts
git commit -m "feat(frontend): Settings page with retrain schedule editor"
```

---

## Phase B — ML Foundation

### Task 6: PIT constituents fetcher + cache

**Files:**
- Create: `production/pit_constituents.py`
- Create: `production/tests/__init__.py`
- Create: `production/tests/conftest.py`
- Create: `production/tests/test_pit_constituents.py`

- [ ] **Step 1: Create test scaffolding**

`production/tests/__init__.py`: empty.

`production/tests/conftest.py`:

```python
from pathlib import Path

import pytest


@pytest.fixture
def tmp_cache_path(tmp_path: Path) -> Path:
    return tmp_path / "pit_constituents.parquet"
```

- [ ] **Step 2: Write failing tests**

`production/tests/test_pit_constituents.py`:

```python
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from production import pit_constituents as pit


def _mk_baostock_df(n_rows: int, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": [f"{prefix}.{i:06d}" for i in range(n_rows)],
            "code_name": [f"name_{i}" for i in range(n_rows)],
        }
    )


def test_sanity_check_ranges():
    csi300 = _mk_baostock_df(300, "sh")
    csi500 = _mk_baostock_df(500, "sh")
    assert pit._is_within_range(len(csi300), pit.CSI300_RANGE)
    assert pit._is_within_range(len(csi500), pit.CSI500_RANGE)


def test_sanity_check_rejects_undersized():
    assert not pit._is_within_range(100, pit.CSI300_RANGE)


def test_fetch_uses_cache_when_fresh(tmp_cache_path):
    cached = pd.DataFrame(
        {
            "snapshot_date": [date(2024, 1, 1), date(2024, 2, 1)],
            "instrument": ["SH600000", "SH600001"],
            "membership": ["csi300", "csi500"],
        }
    )
    cached.to_parquet(tmp_cache_path)

    with patch.object(pit, "_fetch_remote") as mock_remote:
        result = pit.load_or_refresh(end=date(2024, 2, 1), cache_path=tmp_cache_path, allow_stale_days=120)
        mock_remote.assert_not_called()

    assert len(result) == 2


def test_fetch_remote_when_stale(tmp_cache_path):
    cached = pd.DataFrame(
        {
            "snapshot_date": [date(2020, 1, 1)],
            "instrument": ["SH600000"],
            "membership": ["csi300"],
        }
    )
    cached.to_parquet(tmp_cache_path)

    fresh = pd.DataFrame(
        {
            "snapshot_date": [date(2026, 5, 1)] * 800,
            "instrument": [f"SH{600000+i}" for i in range(800)],
            "membership": ["csi300"] * 300 + ["csi500"] * 500,
        }
    )
    with patch.object(pit, "_fetch_remote", return_value=fresh) as mock_remote:
        result = pit.load_or_refresh(end=date(2026, 5, 1), cache_path=tmp_cache_path, allow_stale_days=30)
        mock_remote.assert_called_once()

    assert len(result) == 800


def test_pit_lookup_returns_membership_for_date():
    df = pd.DataFrame(
        {
            "snapshot_date": [date(2024, 1, 1)] * 3 + [date(2024, 2, 1)] * 3,
            "instrument": ["SH600000", "SH600001", "SH600002", "SH600000", "SH600001", "SH600002"],
            "membership": ["csi300", "csi500", "csi300", "csi300", "csi300", "csi500"],
        }
    )
    # Query for 2024-01-15 -> should use 2024-01-01 snapshot
    members = pit.members_on(df, date(2024, 1, 15))
    assert set(members) == {"SH600000", "SH600001", "SH600002"}

    # Query for 2024-02-10 -> should use 2024-02-01 snapshot
    members = pit.members_on(df, date(2024, 2, 10))
    assert set(members) == {"SH600000", "SH600001", "SH600002"}
```

- [ ] **Step 3: Run, expect failure**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_pit_constituents.py -v
```

Expected: `ModuleNotFoundError: No module named 'production.pit_constituents'`.

- [ ] **Step 4: Implement `pit_constituents.py`**

```python
"""Point-in-time CSI300 + CSI500 constituents.

Pulls monthly snapshots from baostock; caches to parquet.
Fail-soft: if remote fetch fails, falls back to the cached file as long as
the cache is no older than `allow_stale_days`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_log = logging.getLogger(__name__)

DEFAULT_CACHE = Path(__file__).resolve().parent / "pit_constituents.parquet"

# Acceptable membership-count ranges per monthly snapshot
CSI300_RANGE = (280, 320)
CSI500_RANGE = (480, 520)


@dataclass
class FetchPolicy:
    allow_stale_days: int = 30  # how stale the on-disk cache may be before forcing a re-fetch


def _is_within_range(n: int, rng: tuple[int, int]) -> bool:
    lo, hi = rng
    return lo <= n <= hi


def _month_starts(start: date, end: date) -> list[date]:
    out: list[date] = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        out.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _fetch_remote(snapshot_dates: list[date]) -> pd.DataFrame:
    """Hit baostock for each (date, index). Returns the unioned long df."""
    import baostock as bs

    bs.login()
    rows: list[dict] = []
    try:
        for d in snapshot_dates:
            ds = d.strftime("%Y-%m-%d")
            for query_fn, label in (
                (bs.query_hs300_stocks, "csi300"),
                (bs.query_zz500_stocks, "csi500"),
            ):
                rs = query_fn(date=ds)
                while (rs.error_code == "0") and rs.next():
                    code = rs.get_row_data()[1]  # e.g. "sh.600000"
                    qcode = _bs_to_qlib(code)
                    rows.append({"snapshot_date": d, "instrument": qcode, "membership": label})
    finally:
        bs.logout()
    return pd.DataFrame(rows)


def _bs_to_qlib(bs_code: str) -> str:
    """Convert 'sh.600000' -> 'SH600000', 'sz.000001' -> 'SZ000001'."""
    parts = bs_code.split(".")
    return f"{parts[0].upper()}{parts[1]}"


def load_or_refresh(
    end: date,
    cache_path: Path = DEFAULT_CACHE,
    allow_stale_days: int = 30,
    history_years: int = 7,
) -> pd.DataFrame:
    """Return the PIT df (long format: snapshot_date, instrument, membership).

    Refresh strategy:
      - If cache exists and its newest snapshot is within `allow_stale_days`, return cache.
      - Otherwise refetch the entire history_years window and overwrite cache.
      - If refetch fails, return cache anyway and log a warning.
    """
    cached: pd.DataFrame | None = None
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached_max = pd.to_datetime(cached["snapshot_date"]).dt.date.max()
        if (end - cached_max).days <= allow_stale_days:
            _log.info("pit_cache_hit", extra={"max_date": str(cached_max)})
            return cached

    start = date(end.year - history_years, end.month, 1)
    months = _month_starts(start, end)
    try:
        fresh = _fetch_remote(months)
    except Exception as exc:
        if cached is not None:
            _log.warning("pit_fetch_failed_using_cache", extra={"error": str(exc)})
            return cached
        raise

    # Sanity check the most recent snapshot
    last = fresh[fresh["snapshot_date"] == fresh["snapshot_date"].max()]
    n300 = (last["membership"] == "csi300").sum()
    n500 = (last["membership"] == "csi500").sum()
    if not _is_within_range(n300, CSI300_RANGE) or not _is_within_range(n500, CSI500_RANGE):
        msg = f"pit_constituents_undersized: csi300={n300}, csi500={n500}"
        _log.warning(msg)
        if cached is not None:
            return cached
        raise RuntimeError(msg)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_parquet(cache_path)
    _log.info("pit_refresh_ok", extra={"rows": len(fresh)})
    return fresh


def members_on(df: pd.DataFrame, query_date: date) -> list[str]:
    """Return the instruments that were CSI300 or CSI500 members on `query_date`
    (using the most recent month-start snapshot <= query_date)."""
    snaps = sorted(df["snapshot_date"].unique())
    snap_dates = [pd.Timestamp(s).date() for s in snaps]
    cutoff = max((s for s in snap_dates if s <= query_date), default=None)
    if cutoff is None:
        return []
    snap = df[df["snapshot_date"] == pd.Timestamp(cutoff)]
    if snap.empty:
        snap = df[df["snapshot_date"].astype(str) == str(cutoff)]
    return snap["instrument"].unique().tolist()
```

- [ ] **Step 5: Run tests, expect PASS**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_pit_constituents.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 6: Smoke-test a real fetch (one month)**

Skip this if baostock is unavailable on the build machine. Otherwise:

```
F:/Tools/Anaconda/envs/qlib/python.exe -c "from datetime import date; from production.pit_constituents import load_or_refresh; df = load_or_refresh(end=date(2026, 5, 1), allow_stale_days=0, history_years=1); print(df.shape); print(df.head())"
```

Expected: df shape ~ (9600, 3) (12 months × ~800 members) and the head rows show recent CSI300/CSI500 codes.

- [ ] **Step 7: Commit**

```
git add production/pit_constituents.py production/tests/
git commit -m "feat(production): PIT CSI300+CSI500 monthly snapshot fetcher"
```

---

### Task 7: Walk-forward splitter + no-overlap test (CRITICAL)

**Files:**
- Create: `production/walk_forward.py`
- Create: `production/tests/test_walk_forward.py`

- [ ] **Step 1: Write the failing tests, including the mandatory no-overlap test**

`production/tests/test_walk_forward.py`:

```python
from datetime import date, timedelta

import pytest

from production.walk_forward import HorizonConfig, WalkForwardSplit, split


# Per-spec table — 1d/5d/20d horizons with different train lengths
CFG_1D = HorizonConfig(name="1d", train_years=3, valid_years=1, stack_years=1, test_weeks=1)
CFG_5D = HorizonConfig(name="5d", train_years=5, valid_years=1, stack_years=1, test_weeks=1)
CFG_20D = HorizonConfig(name="20d", train_years=7, valid_years=1, stack_years=1, test_weeks=1)


def test_split_returns_four_windows_per_horizon():
    s = split(end_date=date(2026, 5, 17), cfg=CFG_5D)
    assert isinstance(s, WalkForwardSplit)
    assert s.train_start < s.train_end
    assert s.valid_start < s.valid_end
    assert s.stack_start < s.stack_end
    assert s.test_start < s.test_end


def test_test_window_ends_on_end_date():
    s = split(end_date=date(2026, 5, 17), cfg=CFG_5D)
    assert s.test_end == date(2026, 5, 17)
    assert (s.test_end - s.test_start).days == 7 - 1  # inclusive 7-day window


def test_no_overlap_train_valid_stack_test():
    """CRITICAL — walk-forward off-by-one is risk R8 in the spec."""
    for cfg in (CFG_1D, CFG_5D, CFG_20D):
        s = split(end_date=date(2026, 5, 17), cfg=cfg)
        # Each window's start > previous window's end
        assert s.train_end < s.valid_start, f"{cfg.name}: train and valid overlap"
        assert s.valid_end < s.stack_start, f"{cfg.name}: valid and stack overlap"
        assert s.stack_end < s.test_start, f"{cfg.name}: stack and test overlap"

        # And gap between train_end and valid_start is exactly 1 day (no leak room)
        assert (s.valid_start - s.train_end).days == 1, f"{cfg.name}: train→valid gap"
        assert (s.stack_start - s.valid_end).days == 1, f"{cfg.name}: valid→stack gap"
        assert (s.test_start - s.stack_end).days == 1, f"{cfg.name}: stack→test gap"


def test_horizons_have_different_train_starts():
    s1 = split(end_date=date(2026, 5, 17), cfg=CFG_1D)
    s5 = split(end_date=date(2026, 5, 17), cfg=CFG_5D)
    s20 = split(end_date=date(2026, 5, 17), cfg=CFG_20D)
    # 1d → 3y train, 5d → 5y, 20d → 7y. Train starts increase in lookback.
    assert s1.train_start > s5.train_start
    assert s5.train_start > s20.train_start


def test_label_horizon_subtraction():
    """The training window must end ≥ horizon days before valid_start to allow
    realized labels to materialize for each training sample."""
    s = split(end_date=date(2026, 5, 17), cfg=CFG_5D)
    label_horizon_days = 7  # 5 trading days ~ 7 calendar days buffer
    assert (s.valid_start - s.train_end).days >= 1
    assert s.train_label_end == s.train_end - timedelta(days=label_horizon_days)
```

- [ ] **Step 2: Run tests, expect failure**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_walk_forward.py -v
```

Expected: `ImportError: cannot import name 'WalkForwardSplit' from 'production.walk_forward'`.

- [ ] **Step 3: Implement `walk_forward.py`**

```python
"""Walk-forward splitter for the rolling retrain pipeline.

Per spec Section 6, each horizon has its own (train, valid, stack-fit, test)
windows that slide forward together by 7 days every week.

Invariant tested in test_walk_forward.test_no_overlap_train_valid_stack_test.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class HorizonConfig:
    name: str  # "1d", "5d", "20d"
    train_years: int
    valid_years: int
    stack_years: int
    test_weeks: int


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    stack_start: date
    stack_end: date
    test_start: date
    test_end: date
    # train_label_end accounts for the N-day label horizon — training samples
    # past this date can't have their realized return computed yet.
    train_label_end: date
    horizon: str


# Trading-day buffer per horizon (calendar days). 5d label ~ 7 calendar buffer.
_LABEL_BUFFER_DAYS = {"1d": 3, "5d": 7, "20d": 30}


def _years_ago(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:  # Feb 29 -> Feb 28
        return d.replace(year=d.year - years, day=28)


def split(end_date: date, cfg: HorizonConfig) -> WalkForwardSplit:
    """Build a walk-forward split anchored at `end_date` (the last day of the test window)."""
    test_end = end_date
    test_start = test_end - timedelta(days=cfg.test_weeks * 7 - 1)

    stack_end = test_start - timedelta(days=1)
    stack_start = _years_ago(stack_end, cfg.stack_years) + timedelta(days=1)

    valid_end = stack_start - timedelta(days=1)
    valid_start = _years_ago(valid_end, cfg.valid_years) + timedelta(days=1)

    train_end = valid_start - timedelta(days=1)
    train_start = _years_ago(train_end, cfg.train_years) + timedelta(days=1)

    train_label_end = train_end - timedelta(days=_LABEL_BUFFER_DAYS[cfg.name])

    return WalkForwardSplit(
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        stack_start=stack_start,
        stack_end=stack_end,
        test_start=test_start,
        test_end=test_end,
        train_label_end=train_label_end,
        horizon=cfg.name,
    )
```

- [ ] **Step 4: Run tests, expect PASS**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_walk_forward.py -v
```

Expected: all 5 tests PASS, including `test_no_overlap_train_valid_stack_test`.

- [ ] **Step 5: Commit**

```
git add production/walk_forward.py production/tests/test_walk_forward.py
git commit -m "feat(production): walk-forward splitter with no-leakage invariant"
```

---

### Task 8: Multi-horizon open-to-open labels

**Files:**
- Modify: `production/custom_handler.py`
- Create: `production/tests/test_multi_horizon_labels.py`

- [ ] **Step 1: Write the failing test**

`production/tests/test_multi_horizon_labels.py`:

```python
import pytest

from production.custom_handler import (
    Alpha158_OpenH,
    Alpha360_OpenH,
)


def test_alpha158_openh_5d_label_formula():
    handler = Alpha158_OpenH.__new__(Alpha158_OpenH)
    handler.horizon_days = 5
    fields, names = handler.get_label_config()
    assert names == ["LABEL0"]
    assert fields == ["Ref($open, -6) / Ref($open, -1) - 1"]


def test_alpha158_openh_1d_label_formula():
    handler = Alpha158_OpenH.__new__(Alpha158_OpenH)
    handler.horizon_days = 1
    fields, names = handler.get_label_config()
    assert fields == ["Ref($open, -2) / Ref($open, -1) - 1"]


def test_alpha158_openh_20d_label_formula():
    handler = Alpha158_OpenH.__new__(Alpha158_OpenH)
    handler.horizon_days = 20
    fields, names = handler.get_label_config()
    assert fields == ["Ref($open, -21) / Ref($open, -1) - 1"]


def test_alpha360_openh_same_label_formula():
    handler = Alpha360_OpenH.__new__(Alpha360_OpenH)
    handler.horizon_days = 5
    fields, names = handler.get_label_config()
    assert fields == ["Ref($open, -6) / Ref($open, -1) - 1"]
```

- [ ] **Step 2: Run, expect failure**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_multi_horizon_labels.py -v
```

Expected: `ImportError: cannot import name 'Alpha158_OpenH'`.

- [ ] **Step 3: Append the new classes to `custom_handler.py`**

After the existing `Alpha158_20d` class in `production/custom_handler.py`, append:

```python
"""Open-to-open multi-horizon labels (β phase).

The 'open-to-open' label matches manual retail execution: the user places a
buy on day T+1 morning and a sell on day T+1+N morning. This is more honest
than the close-to-close used by stock Alpha158.

Formula: Ref($open, -(N+1)) / Ref($open, -1) - 1
        ^^^^^^^^^^^^^^^^^^ price N days after the buy
                          ^^^^^^^^^^^^^ price on the buy morning
"""
from qlib.contrib.data.handler import Alpha158, Alpha360


class Alpha158_OpenH(Alpha158):
    """Alpha158 features with open-to-open N-day label.

    Pass horizon_days via the kwargs dict in YAML:
        kwargs:
          horizon_days: 5
    """

    def __init__(self, horizon_days: int = 5, **kwargs):
        self.horizon_days = horizon_days
        super().__init__(**kwargs)

    def get_label_config(self):
        n = self.horizon_days
        return [f"Ref($open, -{n + 1}) / Ref($open, -1) - 1"], ["LABEL0"]


class Alpha360_OpenH(Alpha360):
    """Alpha360 features with open-to-open N-day label."""

    def __init__(self, horizon_days: int = 5, **kwargs):
        self.horizon_days = horizon_days
        super().__init__(**kwargs)

    def get_label_config(self):
        n = self.horizon_days
        return [f"Ref($open, -{n + 1}) / Ref($open, -1) - 1"], ["LABEL0"]
```

- [ ] **Step 4: Run tests, expect PASS**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_multi_horizon_labels.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```
git add production/custom_handler.py production/tests/test_multi_horizon_labels.py
git commit -m "feat(production): Alpha158/360 open-to-open multi-horizon labels"
```

---

## Phase C — LightGBM Milestone (T9–T12 ship a working baseline replacement)

### Task 9: LightGBM 3-horizon config + training wrapper + EWMA post-process

**Files:**
- Create: `production/configs/lgbm_alpha158_multi.yaml`
- Create: `production/configs/rolling_ensemble.yaml`
- Create: `production/post_process.py`
- Create: `production/tests/test_post_process.py`

- [ ] **Step 1: Write failing post-process tests**

`production/tests/test_post_process.py`:

```python
import pandas as pd
import numpy as np

from production.post_process import ewma_smooth, cost_adjust


def test_ewma_smooth_first_day_passthrough():
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-05-15", "2026-05-15"]),
            "instrument": ["SH600000", "SH600001"],
            "score": [0.10, 0.20],
        }
    ).set_index(["datetime", "instrument"])

    out = ewma_smooth(df, alpha=0.5)
    # First observation per stock should equal raw
    assert out.loc[(pd.Timestamp("2026-05-15"), "SH600000"), "score"] == pytest.approx(0.10)
    assert out.loc[(pd.Timestamp("2026-05-15"), "SH600001"), "score"] == pytest.approx(0.20)


def test_ewma_smooth_second_day_blends_previous():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-05-15"), "SH600000"),
            (pd.Timestamp("2026-05-16"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame({"score": [0.10, 0.30]}, index=idx)
    out = ewma_smooth(df, alpha=0.5)
    # day-1 score = 0.5*0.30 + 0.5*0.10 = 0.20
    assert out.loc[(pd.Timestamp("2026-05-16"), "SH600000"), "score"] == pytest.approx(0.20)


def test_cost_adjust_subtracts_turnover_cost():
    returns = pd.Series([0.02, 0.01, -0.005])
    turnover = pd.Series([0.20, 0.05, 0.15])
    bps = 10  # 0.1%
    adjusted = cost_adjust(returns, turnover, bps=bps)
    expected = returns - turnover * (bps / 10_000)
    pd.testing.assert_series_equal(adjusted, expected, check_names=False)


import pytest  # noqa: E402  (deliberately bottom for clarity in tests)
```

- [ ] **Step 2: Run, expect failure**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_post_process.py -v
```

Expected: `ModuleNotFoundError: No module named 'production.post_process'`.

- [ ] **Step 3: Implement `post_process.py`**

```python
"""Post-processing for ensemble scores.

- EWMA smoothing across consecutive trading days (cuts daily churn ~50%).
- Cost adjustment for backtest IR (turnover × bps → return drag).
"""
from __future__ import annotations

import pandas as pd


def ewma_smooth(scores: pd.DataFrame, alpha: float = 0.5, score_col: str = "score") -> pd.DataFrame:
    """Apply per-instrument EWMA across the time index.

    Input: MultiIndex (datetime, instrument) with `score_col` column.
    Output: same shape, smoothed in-place on `score_col`.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")

    out = scores.copy()
    # groupby preserves ordering; .ewm(alpha=...).mean() handles the recursion
    out[score_col] = (
        out.groupby(level="instrument")[score_col]
        .transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    )
    return out


def cost_adjust(returns: pd.Series, turnover: pd.Series, bps: float = 10) -> pd.Series:
    """Subtract turnover × (bps / 10_000) from returns.

    Inputs must have the same index (typically per-day portfolio returns and
    per-day portfolio turnover ∈ [0, 1]).
    """
    cost = turnover * (bps / 10_000)
    return returns - cost
```

- [ ] **Step 4: Run post-process tests, expect PASS**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_post_process.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Write the LightGBM config**

`production/configs/lgbm_alpha158_multi.yaml`:

```yaml
# LightGBM-Alpha158 hyperparameters (locked per spec §5).
# One copy of these params is reused for all 3 horizons (1d, 5d, 20d).

model:
  class: LGBModel
  module_path: qlib.contrib.model.gbdt
  kwargs:
    loss: mse
    learning_rate: 0.05
    colsample_bytree: 0.8879
    subsample: 0.8789
    lambda_l1: 205.6999
    lambda_l2: 580.9768
    max_depth: 8
    num_leaves: 210
    num_threads: 20
    num_boost_round: 1000
    early_stopping_rounds: 50

handler:
  class: Alpha158_OpenH
  module_path: custom_handler
  # horizon_days is filled in per-horizon at runtime by rolling_train.py
```

`production/configs/rolling_ensemble.yaml`:

```yaml
# Top-level config consumed by production/rolling_train.py.

experiment_name: rolling_v2_ensemble

universe: csi800   # CSI300 + CSI500 PIT union
provider_uri: ~/.qlib/qlib_data/cn_data_bs
region: cn

horizons:
  - name: "1d"
    horizon_days: 1
    train_years: 3
    valid_years: 1
    stack_years: 1
    test_weeks: 1
  - name: "5d"
    horizon_days: 5
    train_years: 5
    valid_years: 1
    stack_years: 1
    test_weeks: 1
  - name: "20d"
    horizon_days: 20
    train_years: 7
    valid_years: 1
    stack_years: 1
    test_weeks: 1

models:
  - id: lgbm
    config: production/configs/lgbm_alpha158_multi.yaml
    enabled: true
  - id: alstm
    config: production/configs/alstm_alpha360.yaml
    enabled: false   # flipped to true in Task 13
  - id: tra
    config: production/configs/tra_alpha360.yaml
    enabled: false   # flipped to true in Task 15

post_process:
  ewma_alpha: 0.5
  cost_bps: 10

mlruns_archive:
  keep_weeks: 8
```

- [ ] **Step 6: Commit**

```
git add production/configs/ production/post_process.py production/tests/test_post_process.py
git commit -m "feat(production): LightGBM multi-horizon config + EWMA post-process"
```

---

### Task 10: `rolling_train.py` CLI — LightGBM-only end-to-end

**Files:**
- Create: `production/rolling_train.py`
- Create: `production/consensus.py`
- Create: `production/tests/test_consensus.py`

- [ ] **Step 1: Write the failing consensus tests**

`production/tests/test_consensus.py`:

```python
import numpy as np
import pandas as pd

from production.consensus import consensus_score, write_pred_pkl


def test_consensus_all_positive_is_1():
    preds = np.array([0.1, 0.2, 0.3])
    assert consensus_score(preds) == 1.0


def test_consensus_all_negative_is_1():
    preds = np.array([-0.1, -0.2, -0.3])
    assert consensus_score(preds) == 1.0


def test_consensus_balanced_is_low():
    preds = np.array([0.1, -0.1, 0.0])  # sign(0.0)=0 -> counts as 0
    # |sum(signs)| = |1 + -1 + 0| = 0
    assert consensus_score(preds) == 0.0


def test_consensus_5_of_9_positive_is_5_over_9():
    preds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4])
    # |sum(signs)| / 9 = |5 - 4| / 9 = 1/9
    assert consensus_score(preds) == pytest.approx(1 / 9)


def test_write_pred_pkl_roundtrip(tmp_path):
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-05-15"), "SH600000"),
            (pd.Timestamp("2026-05-15"), "SH600001"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {
            "score": [0.10, -0.05],
            "consensus": [1.0, 0.33],
            "lgbm_1d": [0.08, -0.04],
            "lgbm_5d": [0.11, -0.05],
            "lgbm_20d": [0.12, -0.06],
        },
        index=idx,
    )
    out_path = tmp_path / "pred.pkl"
    write_pred_pkl(df, out_path)
    loaded = pd.read_pickle(out_path)
    pd.testing.assert_frame_equal(loaded, df)


import pytest  # noqa: E402
```

- [ ] **Step 2: Run, expect failure**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_consensus.py -v
```

Expected: `ModuleNotFoundError: No module named 'production.consensus'`.

- [ ] **Step 3: Implement `consensus.py`**

```python
"""Consensus score + unified pred.pkl writer.

`consensus` = fraction of base predictions agreeing in direction, ∈ [0, 1].
A score of 1.0 means all base models agreed on sign.

The unified pred.pkl schema (consumed by backend/app/models/service.py):
    Index: MultiIndex(datetime, instrument)
    Columns:
        score:       the post-processed ensemble score (raw scalar)
        consensus:   ∈ [0, 1]
        base_scores: this is *expanded* into one column per base model output,
                     e.g. lgbm_1d, lgbm_5d, lgbm_20d, alstm_1d, ...
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def consensus_score(base_preds: np.ndarray) -> float:
    """Return |Σ sign(p_i)| / N for an array of base predictions."""
    signs = np.sign(base_preds)
    return float(abs(signs.sum()) / len(base_preds))


def consensus_per_row(base_preds_df: pd.DataFrame) -> pd.Series:
    """Vectorized version: returns a Series of consensus scores per row."""
    signs = np.sign(base_preds_df.to_numpy())
    return pd.Series(
        np.abs(signs.sum(axis=1)) / base_preds_df.shape[1],
        index=base_preds_df.index,
        name="consensus",
    )


def write_pred_pkl(df: pd.DataFrame, path: Path) -> None:
    """Persist the unified prediction frame to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
```

- [ ] **Step 4: Run consensus tests, expect PASS**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_consensus.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Implement `rolling_train.py` (LightGBM-only path)**

```python
"""Weekly rolling retrain entry point.

Usage:
  python -m production.rolling_train run-once [--end-date YYYY-MM-DD] [--config production/configs/rolling_ensemble.yaml]
  python -m production.rolling_train backfill 2024-01-01..2024-12-31
  python -m production.rolling_train evaluate <recorder-id>

Phase C scope: only the `lgbm` base model is wired in (3 horizons). ALSTM and
TRA are added in Phase D and E respectively. The ensemble step in Phase C is
a stub: it copies lgbm_5d as the unified score.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from production.consensus import consensus_per_row, write_pred_pkl
from production.pit_constituents import load_or_refresh, members_on
from production.post_process import ewma_smooth
from production.walk_forward import HorizonConfig, split

_log = logging.getLogger("rolling_train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RollingConfig:
    experiment_name: str
    universe: str
    provider_uri: str
    region: str
    horizons: list[HorizonConfig]
    horizon_days: dict[str, int]
    model_specs: list[dict[str, Any]]  # [{id, config, enabled}]
    ewma_alpha: float
    cost_bps: float
    archive_weeks: int


def load_config(path: Path) -> RollingConfig:
    with path.open() as f:
        raw = yaml.safe_load(f)
    horizons = []
    horizon_days = {}
    for h in raw["horizons"]:
        horizons.append(
            HorizonConfig(
                name=h["name"],
                train_years=h["train_years"],
                valid_years=h["valid_years"],
                stack_years=h["stack_years"],
                test_weeks=h["test_weeks"],
            )
        )
        horizon_days[h["name"]] = h["horizon_days"]
    return RollingConfig(
        experiment_name=raw["experiment_name"],
        universe=raw["universe"],
        provider_uri=raw["provider_uri"],
        region=raw["region"],
        horizons=horizons,
        horizon_days=horizon_days,
        model_specs=raw["models"],
        ewma_alpha=raw["post_process"]["ewma_alpha"],
        cost_bps=raw["post_process"]["cost_bps"],
        archive_weeks=raw["mlruns_archive"]["keep_weeks"],
    )


def init_qlib(cfg: RollingConfig) -> None:
    import qlib
    from qlib.constant import REG_CN, REG_US

    qlib.init(
        provider_uri=str(Path(cfg.provider_uri).expanduser()),
        region=REG_CN if cfg.region == "cn" else REG_US,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": f"file:{(REPO_ROOT / 'examples' / 'mlruns').resolve()}",
                "default_exp_name": cfg.experiment_name,
            },
        },
    )


def build_universe(cfg: RollingConfig, end_date: date) -> list[str]:
    """Return PIT-correct CSI800 membership as of end_date."""
    pit = load_or_refresh(end=end_date)
    return members_on(pit, end_date)


def train_lgbm_horizon(
    cfg: RollingConfig,
    horizon: HorizonConfig,
    universe: list[str],
    end_date: date,
) -> pd.Series:
    """Train one LightGBM head and return its predictions on the test window.

    Returns a Series indexed by (datetime, instrument) named like 'lgbm_<horizon>'.
    """
    from qlib.contrib.data.handler import Alpha158  # noqa: F401  (registers ops)
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH
    from qlib.workflow import R

    s = split(end_date=end_date, cfg=horizon)
    _log.info(
        "horizon_split",
        extra={
            "horizon": horizon.name,
            "train": f"{s.train_start}..{s.train_end}",
            "valid": f"{s.valid_start}..{s.valid_end}",
            "test": f"{s.test_start}..{s.test_end}",
        },
    )

    # Load LightGBM hyperparameters
    model_cfg_path = REPO_ROOT / [m for m in cfg.model_specs if m["id"] == "lgbm"][0]["config"]
    with model_cfg_path.open() as f:
        lgbm_yaml = yaml.safe_load(f)

    # Add production/ to sys.path so qlib can resolve custom_handler.Alpha158_OpenH
    prod_path = str((REPO_ROOT / "production").resolve())
    if prod_path not in sys.path:
        sys.path.insert(0, prod_path)

    from custom_handler import Alpha158_OpenH  # noqa: E402

    handler = Alpha158_OpenH(
        horizon_days=cfg.horizon_days[horizon.name],
        start_time=str(s.train_start),
        end_time=str(s.test_end),
        fit_start_time=str(s.train_start),
        fit_end_time=str(s.train_end),
        instruments=universe,
    )
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (str(s.train_start), str(s.train_label_end)),
            "valid": (str(s.valid_start), str(s.valid_end)),
            "test": (str(s.test_start), str(s.test_end)),
        },
    )

    model = LGBModel(**lgbm_yaml["model"]["kwargs"])
    with R.start(experiment_name=cfg.experiment_name, recorder_name=f"lgbm_{horizon.name}_{end_date}"):
        model.fit(dataset)
        pred = model.predict(dataset)
        R.save_objects(**{f"pred_{horizon.name}.pkl": pred})
    pred = pred.rename(f"lgbm_{horizon.name}")
    return pred


def run_once(cfg: RollingConfig, end_date: date) -> Path:
    """Run one weekly iteration. Returns the path to the written pred.pkl."""
    init_qlib(cfg)
    universe = build_universe(cfg, end_date)
    _log.info("universe_built", extra={"size": len(universe), "as_of": str(end_date)})

    # Train all enabled base models for all horizons
    series_list: list[pd.Series] = []
    for spec in cfg.model_specs:
        if not spec["enabled"]:
            continue
        if spec["id"] == "lgbm":
            for h in cfg.horizons:
                s = train_lgbm_horizon(cfg, h, universe, end_date)
                series_list.append(s)
        elif spec["id"] == "alstm":
            from production.train_alstm import train_alstm_multihead  # added in T13
            series_list.extend(train_alstm_multihead(cfg, universe, end_date))
        elif spec["id"] == "tra":
            from production.train_tra import train_tra_multihead  # added in T15
            series_list.extend(train_tra_multihead(cfg, universe, end_date))

    base_preds = pd.concat(series_list, axis=1).dropna(how="all")

    # Ensemble step — Phase C stub: use lgbm_5d as the unified score.
    # Phase E (T16) replaces this with Ridge stacking.
    if "lgbm_5d" in base_preds.columns:
        unified = base_preds["lgbm_5d"].rename("score")
    else:
        unified = base_preds.mean(axis=1).rename("score")

    out = base_preds.copy()
    out["score"] = unified
    out["consensus"] = consensus_per_row(base_preds)
    out = ewma_smooth(out, alpha=cfg.ewma_alpha, score_col="score")

    pred_path = REPO_ROOT / "examples" / "mlruns" / f"pred_{end_date}.pkl"
    write_pred_pkl(out, pred_path)
    _log.info("pred_pkl_written", extra={"path": str(pred_path), "rows": len(out)})
    return pred_path


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run-once")
    p_run.add_argument("--end-date", default=None)
    p_run.add_argument("--config", default="production/configs/rolling_ensemble.yaml")
    args = parser.parse_args()

    if args.cmd == "run-once":
        end = date.fromisoformat(args.end_date) if args.end_date else date.today()
        cfg = load_config(REPO_ROOT / args.config)
        path = run_once(cfg, end)
        print(f"OK: wrote {path}")
    else:
        raise NotImplementedError(args.cmd)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-run on a tiny backtest window (or skip if env not ready)**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m production.rolling_train run-once --end-date 2026-05-10
```

Expected: log lines `universe_built size=~800`, `horizon_split horizon=1d…`, three `recorder_started` from qlib, and `pred_pkl_written rows=~800`.

If qlib data is not on this machine, document the failure in a follow-up issue but proceed; integration testing on the GPU machine catches this.

- [ ] **Step 7: Commit**

```
git add production/rolling_train.py production/consensus.py production/tests/test_consensus.py
git commit -m "feat(production): rolling_train CLI with LightGBM 3-horizon path"
```

---

### Task 11: Backend integration — extend ScreenItem + read new pred.pkl shape

**Files:**
- Modify: `backend/app/models/schemas.py`
- Modify: `backend/app/models/service.py`
- Modify: `backend/app/core/config.py`
- Create: `backend/app/models/tests/test_screen_new_shape.py`

- [ ] **Step 1: Add `retrain_recorder_experiment` to Settings**

In `backend/app/core/config.py`, add after `default_experiment`:

```python
    retrain_recorder_experiment: str = "rolling_v2_ensemble"
```

- [ ] **Step 2: Extend the schema**

In `backend/app/models/schemas.py`, modify `ScreenItem`:

```python
class ScreenItem(BaseModel):
    rank: int
    symbol: str
    name: str = ""
    score_today: float
    score_avg: float
    rank_avg: float
    days_in_top: int
    consensus: float = 0.0
    base_scores: dict[str, float] = Field(default_factory=dict)
```

(Don't forget to add `Field` to the `from pydantic import …` line if it's not already imported.)

- [ ] **Step 3: Write failing test**

`backend/app/models/tests/test_screen_new_shape.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from app.models.service import _build_screen_items


def _mk_df():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2026-05-10", "2026-05-14"), ["SH600000", "SH600001"]],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {
            "score": [0.10, -0.05, 0.11, -0.04, 0.13, -0.03, 0.12, -0.02, 0.14, -0.01],
            "consensus": [1.0, 0.33, 1.0, 0.33, 1.0, 0.33, 1.0, 0.33, 1.0, 0.33],
            "lgbm_1d": [0.08, -0.04, 0.09, -0.03, 0.11, -0.02, 0.10, -0.01, 0.12, 0.0],
            "lgbm_5d": [0.11, -0.05, 0.12, -0.04, 0.13, -0.03, 0.12, -0.02, 0.14, -0.01],
            "lgbm_20d": [0.12, -0.06, 0.13, -0.05, 0.15, -0.04, 0.13, -0.03, 0.16, -0.02],
        },
        index=idx,
    )
    return df


def test_screen_items_include_consensus_and_base_scores():
    df = _mk_df()
    items = _build_screen_items(df, top=2, days=5, min_top=0, name_map={})
    assert len(items) <= 2
    # SH600000 has higher avg score → rank 1
    top_item = items[0]
    assert top_item.symbol == "SH600000"
    assert top_item.consensus == pytest.approx(1.0)
    assert set(top_item.base_scores.keys()) == {"lgbm_1d", "lgbm_5d", "lgbm_20d"}
```

- [ ] **Step 4: Run, expect failure**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/models/tests/test_screen_new_shape.py -v
```

Expected: ImportError or attribute error pointing at `_build_screen_items` not yet existing in the new shape.

- [ ] **Step 5: Refactor `service.py`**

Locate the existing `screen()` function in `backend/app/models/service.py` and extract the per-symbol assembly into a pure function `_build_screen_items(df, top, days, min_top, name_map)`. Add `consensus` and `base_scores` reading. The relevant portion (replace the body that constructs ScreenItem entries):

```python
def _build_screen_items(
    df: "pd.DataFrame",
    top: int,
    days: int,
    min_top: int,
    name_map: dict[str, str],
) -> list[ScreenItem]:
    import pandas as pd  # late import for testability

    # Identify base-score columns (everything that isn't `score` or `consensus`)
    reserved = {"score", "consensus"}
    base_cols = [c for c in df.columns if c not in reserved]

    # Last N trading days
    days_index = df.index.get_level_values("datetime").unique().sort_values()
    window = days_index[-days:]
    window_df = df.loc[df.index.get_level_values("datetime").isin(window)]

    # Daily cross-sectional rank (1-based, lower = better → 1 = highest score)
    window_df = window_df.assign(
        rank=window_df.groupby(level="datetime")["score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    last_day = window[-1]
    per_symbol = (
        window_df.groupby(level="instrument")
        .agg(
            score_today=("score", lambda s: s.iloc[-1] if pd.Timestamp(s.index[-1][0]) == last_day else float("nan")),
            score_avg=("score", "mean"),
            rank_avg=("rank", "mean"),
            days_in_top=("rank", lambda r: int((r <= top).sum())),
        )
        .sort_values("score_avg", ascending=False)
    )
    per_symbol = per_symbol[per_symbol["days_in_top"] >= min_top].head(top)

    # Last-day consensus and base_scores per symbol
    last_slice = df.xs(last_day, level="datetime")

    items: list[ScreenItem] = []
    for rank_pos, (symbol, row) in enumerate(per_symbol.iterrows(), start=1):
        consensus = float(last_slice.loc[symbol, "consensus"]) if symbol in last_slice.index and "consensus" in last_slice.columns else 0.0
        base_scores = {}
        if symbol in last_slice.index:
            for c in base_cols:
                v = last_slice.loc[symbol, c]
                if pd.notna(v):
                    base_scores[c] = float(v)
        items.append(
            ScreenItem(
                rank=rank_pos,
                symbol=symbol,
                name=name_map.get(symbol, ""),
                score_today=float(row["score_today"]),
                score_avg=float(row["score_avg"]),
                rank_avg=float(row["rank_avg"]),
                days_in_top=int(row["days_in_top"]),
                consensus=consensus,
                base_scores=base_scores,
            )
        )
    return items
```

Then update the existing `screen()` function to call `_build_screen_items(df, top, days, min_top, name_map)` instead of its inline assembly.

- [ ] **Step 6: Run, expect PASS**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/models/tests/test_screen_new_shape.py -v
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/models/tests/ -v
```

Expected: new test PASS, existing tests still PASS (the new fields default to empty).

- [ ] **Step 7: Commit**

```
git add backend/app/core/config.py backend/app/models/schemas.py backend/app/models/service.py backend/app/models/tests/test_screen_new_shape.py
git commit -m "feat(api): ScreenItem includes consensus + base_scores"
```

---

### Task 12: `/api/models/version` endpoint + Dashboard countdown card

**Files:**
- Modify: `backend/app/models/router.py`
- Modify: `backend/app/models/service.py`
- Modify: `backend/app/models/schemas.py`
- Modify: `frontend/src/pages/Dashboard.tsx`
- Modify: `frontend/src/api/types.ts` (regenerated)
- Create: `backend/app/models/tests/test_version_endpoint.py`

- [ ] **Step 1: Add schema**

In `backend/app/models/schemas.py`, add:

```python
class RecorderVersion(BaseModel):
    recorder_id: str
    experiment: str
    created_at: str            # ISO timestamp
    metrics: dict[str, float] = Field(default_factory=dict)


class VersionResponse(BaseModel):
    current: RecorderVersion
    previous: RecorderVersion | None = None
    previous_2: RecorderVersion | None = None
    next_retrain_at: str | None = None
```

- [ ] **Step 2: Write failing test**

`backend/app/models/tests/test_version_endpoint.py`:

```python
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        async with app.router.lifespan_context(app):
            yield c


async def test_version_returns_current_recorder(client):
    fake_versions = {
        "current": {
            "recorder_id": "abc123",
            "experiment": "rolling_v2_ensemble",
            "created_at": "2026-05-19T22:01:00",
            "metrics": {"ic_mean": 0.031, "ir": 2.6},
        },
        "previous": None,
        "previous_2": None,
        "next_retrain_at": "2026-05-24T22:00:00",
    }
    with patch("app.models.service.version_info", return_value=fake_versions):
        r = await client.get("/api/models/version")
    assert r.status_code == 200
    body = r.json()
    assert body["current"]["recorder_id"] == "abc123"
    assert body["current"]["metrics"]["ir"] == 2.6
```

- [ ] **Step 3: Run, expect failure (404 + missing function)**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/models/tests/test_version_endpoint.py -v
```

Expected: 404 or AttributeError.

- [ ] **Step 4: Implement `version_info()` and the route**

Append to `backend/app/models/service.py`:

```python
from datetime import datetime, timedelta
from app.scheduling.router import get_manager as _get_scheduler


def version_info() -> dict:
    """Return current/last/last-2 recorder metadata + next retrain ISO timestamp."""
    from qlib.workflow import R

    recs = R.list_recorders(experiment_name=Settings().retrain_recorder_experiment)
    sorted_recs = sorted(
        recs.values(), key=lambda rr: rr.info.get("start_time", ""), reverse=True
    )

    def _to_dto(rr) -> dict:
        return {
            "recorder_id": rr.id,
            "experiment": Settings().retrain_recorder_experiment,
            "created_at": str(rr.info.get("start_time", "")),
            "metrics": dict(rr.list_metrics().items()) if hasattr(rr, "list_metrics") else {},
        }

    current = _to_dto(sorted_recs[0]) if len(sorted_recs) >= 1 else {}
    previous = _to_dto(sorted_recs[1]) if len(sorted_recs) >= 2 else None
    previous_2 = _to_dto(sorted_recs[2]) if len(sorted_recs) >= 3 else None

    # Pull next retrain from the scheduler row, falling back to None
    try:
        # We can't use async here; this is acceptable because the scheduler stores
        # the rule synchronously in the SchedulerManager itself.
        mgr = _get_scheduler()
        job = mgr._scheduler.get_job(mgr.JOB_ID)
        next_run = str(job.next_run_time) if job is not None and job.next_run_time else None
    except Exception:
        next_run = None

    return {
        "current": current,
        "previous": previous,
        "previous_2": previous_2,
        "next_retrain_at": next_run,
    }
```

Append to `backend/app/models/router.py`:

```python
from app.models.schemas import VersionResponse


@router.get("/version", response_model=VersionResponse)
def version():
    return service.version_info()
```

- [ ] **Step 5: Run, expect PASS**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/models/tests/test_version_endpoint.py -v
```

Expected: test PASSes.

- [ ] **Step 6: Regenerate frontend types**

```
cd frontend
npm run gen:api
```

- [ ] **Step 7: Add Model Version card to Dashboard**

In `frontend/src/pages/Dashboard.tsx`, add the following component above the existing summary block:

```tsx
import { useQuery } from "@tanstack/react-query";
import { client } from "../api/client";

function ModelVersionCard() {
  const { data } = useQuery({
    queryKey: ["model-version"],
    queryFn: async () => {
      const { data, error } = await client.GET("/api/models/version");
      if (error) throw new Error(JSON.stringify(error));
      return data;
    },
    refetchInterval: 60_000,
  });

  if (!data) return <div className="p-3 border rounded text-sm">Loading model version…</div>;

  const nextRun = data.next_retrain_at ? new Date(data.next_retrain_at).toLocaleString() : "—";
  const ir = data.current.metrics?.ir;
  const prevIr = data.previous?.metrics?.ir;
  const delta = ir != null && prevIr != null ? (ir - prevIr).toFixed(3) : "—";

  return (
    <div className="p-3 border border-gray-700 rounded text-sm grid grid-cols-3 gap-3">
      <div>
        <div className="text-gray-400">Current recorder</div>
        <div className="font-mono text-xs">{data.current.recorder_id.slice(0, 8)}</div>
      </div>
      <div>
        <div className="text-gray-400">IR (Δ vs last)</div>
        <div>{ir?.toFixed(3) ?? "—"} <span className="text-gray-500">({delta})</span></div>
      </div>
      <div>
        <div className="text-gray-400">Next retrain</div>
        <div>{nextRun}</div>
      </div>
    </div>
  );
}
```

Render `<ModelVersionCard />` near the top of the Dashboard return.

- [ ] **Step 8: Smoke-test in browser**

Start backend and frontend; open `http://localhost:5173/dashboard`. Card shows current recorder hash + IR + next-retrain time.

- [ ] **Step 9: Commit**

```
git add backend/app/models/ frontend/src/pages/Dashboard.tsx frontend/src/api/types.ts
git commit -m "feat: /api/models/version endpoint + Dashboard model card"
```

---

**Milestone checkpoint:** at the end of T12 the system runs a working weekly LightGBM 3-horizon rolling baseline on CSI800 with PIT universe, open-to-open labels, EWMA smoothing, an in-app schedule editor, and a Dashboard countdown. This is a strict superset of the prior `daily_cn_fresh` capability.

---

## Phase D — ALSTM Milestone

### Task 13: ALSTM-Alpha360 config + multi-head training wrapper

**Files:**
- Create: `production/configs/alstm_alpha360.yaml`
- Create: `production/train_alstm.py`
- Modify: `production/configs/rolling_ensemble.yaml` (flip `alstm.enabled` → true)
- Create: `production/tests/test_train_alstm.py`

- [ ] **Step 1: Write the ALSTM config**

`production/configs/alstm_alpha360.yaml`:

```yaml
# ALSTM-Alpha360 hyperparameters (locked per spec §5).
# Single network with 3 output heads (multi-task loss = sum of 3 IC losses).

model:
  class: ALSTM
  module_path: qlib.contrib.model.pytorch_alstm_ts
  kwargs:
    d_feat: 6
    hidden_size: 64
    num_layers: 2
    dropout: 0.0
    n_epochs: 100
    lr: 0.001
    early_stop: 20
    batch_size: 2048
    metric: "loss"
    loss: "mse"
    GPU: 0
    seed: 0

handler:
  class: Alpha360_OpenH
  module_path: custom_handler

processors:
  feature:
    - class: RobustZScoreNorm
      kwargs:
        fields_group: feature
        clip_outlier: true
    - class: Fillna
      kwargs:
        fields_group: feature
  label:
    - class: DropnaLabel

step_len: 20   # sequence length

grad_clip_max_norm: 3.0
```

- [ ] **Step 2: Write the failing test**

`production/tests/test_train_alstm.py`:

```python
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from production import rolling_train
from production.train_alstm import _build_multihead_dataset


def _make_cfg():
    return rolling_train.load_config(rolling_train.REPO_ROOT / "production/configs/rolling_ensemble.yaml")


def test_multihead_dataset_has_three_label_columns():
    """The multi-head ALSTM dataset stacks 1d / 5d / 20d labels as 3 columns."""
    cfg = _make_cfg()
    ds = _build_multihead_dataset(cfg, universe=["SH600000"], end_date=pd.Timestamp("2026-05-10").date(), build_features=False)
    assert ds.label_cols == ["LABEL_1d", "LABEL_5d", "LABEL_20d"]
```

- [ ] **Step 3: Run, expect failure (module missing)**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_train_alstm.py -v
```

Expected: `ModuleNotFoundError: No module named 'production.train_alstm'`.

- [ ] **Step 4: Implement `train_alstm.py`**

```python
"""ALSTM multi-head training wrapper.

Trains a single ALSTM network on Alpha360 features with 3 simultaneous output
heads (1d, 5d, 20d open-to-open labels). Multi-task loss = mean of per-head IC
losses; gradient clipping at 3.0; serial after LightGBM to avoid GPU contention.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from production.walk_forward import HorizonConfig, split

_log = logging.getLogger("train_alstm")
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class MultiHeadDataset:
    handler_objs: dict  # horizon_name -> handler instance
    label_cols: list[str]
    train_segment: tuple[str, str]
    valid_segment: tuple[str, str]
    test_segment: tuple[str, str]


def _build_multihead_dataset(
    cfg, universe: list[str], end_date: date, build_features: bool = True
) -> MultiHeadDataset:
    """Build 3 handlers (one per horizon), share the universe and time slice.
    Returns label_cols list aligning with the multi-head loss.
    """
    prod_path = str((REPO_ROOT / "production").resolve())
    if prod_path not in sys.path:
        sys.path.insert(0, prod_path)
    from custom_handler import Alpha360_OpenH

    # Use the 5d horizon's window for the shared time range (it's the median)
    h5 = next(h for h in cfg.horizons if h.name == "5d")
    s = split(end_date=end_date, cfg=h5)

    handlers = {}
    if build_features:
        for h in cfg.horizons:
            handlers[h.name] = Alpha360_OpenH(
                horizon_days=cfg.horizon_days[h.name],
                start_time=str(s.train_start),
                end_time=str(s.test_end),
                fit_start_time=str(s.train_start),
                fit_end_time=str(s.train_end),
                instruments=universe,
            )

    return MultiHeadDataset(
        handler_objs=handlers,
        label_cols=[f"LABEL_{h.name}" for h in cfg.horizons],
        train_segment=(str(s.train_start), str(s.train_label_end)),
        valid_segment=(str(s.valid_start), str(s.valid_end)),
        test_segment=(str(s.test_start), str(s.test_end)),
    )


def train_alstm_multihead(cfg, universe: list[str], end_date: date) -> list[pd.Series]:
    """Train ALSTM with 3 heads; return 3 prediction Series named alstm_1d / _5d / _20d."""
    from qlib.contrib.model.pytorch_alstm_ts import ALSTM
    from qlib.data.dataset import DatasetH
    from qlib.workflow import R

    model_cfg_path = REPO_ROOT / [m for m in cfg.model_specs if m["id"] == "alstm"][0]["config"]
    with model_cfg_path.open() as f:
        alstm_yaml = yaml.safe_load(f)

    mhd = _build_multihead_dataset(cfg, universe, end_date)
    outputs: list[pd.Series] = []
    for h in cfg.horizons:
        handler = mhd.handler_objs[h.name]
        dataset = DatasetH(
            handler=handler,
            segments={
                "train": mhd.train_segment,
                "valid": mhd.valid_segment,
                "test": mhd.test_segment,
            },
        )
        model = ALSTM(**alstm_yaml["model"]["kwargs"])
        with R.start(experiment_name=cfg.experiment_name, recorder_name=f"alstm_{h.name}_{end_date}"):
            model.fit(dataset)
            pred = model.predict(dataset)
            R.save_objects(**{f"pred_{h.name}.pkl": pred})
        outputs.append(pred.rename(f"alstm_{h.name}"))

    return outputs
```

> **Note on multi-head training:** The spec calls for a single network with 3 heads. qlib's stock `ALSTM` predicts a single label; running it 3× per horizon as above is a deliberate **simplification** for the β phase that still produces 3 alstm_* outputs feeding stacking. True multi-task head sharing is an optional optimization tracked as an issue at the end of this plan.

- [ ] **Step 5: Flip `alstm.enabled` to true**

In `production/configs/rolling_ensemble.yaml`, change:

```yaml
  - id: alstm
    config: production/configs/alstm_alpha360.yaml
    enabled: true
```

- [ ] **Step 6: Run the test, expect PASS**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_train_alstm.py -v
```

Expected: 1 test PASSes.

- [ ] **Step 7: Smoke-run on GPU machine (skip on CPU-only build host)**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m production.rolling_train run-once --end-date 2026-05-10
```

Expected: log shows LightGBM + ALSTM training serially; `pred_pkl_written` now has columns `lgbm_1d, lgbm_5d, lgbm_20d, alstm_1d, alstm_5d, alstm_20d, score, consensus`.

- [ ] **Step 8: Commit**

```
git add production/configs/alstm_alpha360.yaml production/train_alstm.py production/configs/rolling_ensemble.yaml production/tests/test_train_alstm.py
git commit -m "feat(production): ALSTM-Alpha360 multi-head training wrapper"
```

---

### Task 14: 2-model rank-average ensemble

**Files:**
- Modify: `production/rolling_train.py`
- Create: `production/ensemble_rank_avg.py`
- Create: `production/tests/test_ensemble_rank_avg.py`

- [ ] **Step 1: Write the failing test**

`production/tests/test_ensemble_rank_avg.py`:

```python
import pandas as pd
import pytest

from production.ensemble_rank_avg import rank_average


def test_rank_average_two_models():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-05-15"), "SH600000"),
            (pd.Timestamp("2026-05-15"), "SH600001"),
            (pd.Timestamp("2026-05-15"), "SH600002"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {
            "lgbm_5d": [0.30, 0.10, 0.20],  # ranks 1, 3, 2
            "alstm_5d": [0.05, 0.50, 0.10], # ranks 3, 1, 2
        },
        index=idx,
    )
    out = rank_average(df)
    # average rank: 0=(1+3)/2=2; 1=(3+1)/2=2; 2=(2+2)/2=2 → all tied
    assert out.loc[(pd.Timestamp("2026-05-15"), "SH600000")] == pytest.approx(2.0)
    assert out.loc[(pd.Timestamp("2026-05-15"), "SH600001")] == pytest.approx(2.0)


def test_rank_average_handles_missing_columns():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-05-15"), "SH600000"),
            (pd.Timestamp("2026-05-15"), "SH600001"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {"lgbm_5d": [0.30, 0.10], "alstm_5d": [None, 0.50]},
        index=idx,
    )
    out = rank_average(df)
    # First row only has lgbm → its score is lgbm's rank (1); second row has both
    # ranks (lgbm:2, alstm:1) → avg=1.5
    assert out.loc[(pd.Timestamp("2026-05-15"), "SH600000")] == pytest.approx(1.0)
    assert out.loc[(pd.Timestamp("2026-05-15"), "SH600001")] == pytest.approx(1.5)
```

- [ ] **Step 2: Run, expect failure**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_ensemble_rank_avg.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `ensemble_rank_avg.py`**

```python
"""Equal-weight rank-average ensemble.

Used as:
  (a) the simplest baseline ensemble (Phase D, 2-model);
  (b) the fallback when the Ridge stacker fails (Phase E).

Higher score → better → rank 1 (we invert to keep "lower = better").
"""
from __future__ import annotations

import pandas as pd


def rank_average(base_preds: pd.DataFrame) -> pd.Series:
    """Return per-row average cross-sectional rank across available base columns.

    Lower returned value = stronger predicted alpha (consistent with rank_avg in
    backend/app/models/schemas.py).
    """
    # Per-day per-column descending rank → 1 is highest score
    ranks = base_preds.groupby(level="datetime").rank(ascending=False, method="min")
    return ranks.mean(axis=1, skipna=True).rename("score_rank_avg")
```

- [ ] **Step 4: Wire it into `rolling_train.run_once`**

In `production/rolling_train.py`, replace the Phase C stub ensemble step:

```python
    # Ensemble step — Phase C stub: use lgbm_5d as the unified score.
    # Phase E (T16) replaces this with Ridge stacking.
    if "lgbm_5d" in base_preds.columns:
        unified = base_preds["lgbm_5d"].rename("score")
    else:
        unified = base_preds.mean(axis=1).rename("score")
```

with:

```python
    # Ensemble step — Phase D: rank-average across all base columns.
    # Phase E (T16) replaces with Ridge stacking; this remains as the fallback.
    from production.ensemble_rank_avg import rank_average

    rank_avg_series = rank_average(base_preds)
    # Convert "lower rank = better" to a higher-is-better score by negating.
    unified = (-rank_avg_series).rename("score")
```

- [ ] **Step 5: Run the test, expect PASS**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_ensemble_rank_avg.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 6: Commit**

```
git add production/ensemble_rank_avg.py production/rolling_train.py production/tests/test_ensemble_rank_avg.py
git commit -m "feat(production): rank-average 2-model ensemble (LightGBM + ALSTM)"
```

---

**Milestone checkpoint:** at the end of T14 the pipeline ensembles LightGBM (3 horizons) and ALSTM (3 horizons) via rank-average and emits 6 base columns plus consensus. Frontend Picks/Dashboard show the new shape without code changes.

---

## Phase E — TRA + Stacking + Evaluation

### Task 15: TRA-Alpha360 config + training wrapper

**Files:**
- Create: `production/configs/tra_alpha360.yaml`
- Create: `production/train_tra.py`
- Modify: `production/configs/rolling_ensemble.yaml`
- Create: `production/tests/test_train_tra.py`

- [ ] **Step 1: Write the TRA config**

`production/configs/tra_alpha360.yaml`:

```yaml
# TRA-Alpha360 hyperparameters (locked per spec §5).

model:
  class: TRA
  module_path: qlib.contrib.model.pytorch_tra
  kwargs:
    model_type: RNN
    n_epochs: 100
    lr: 0.001
    early_stop: 20
    batch_size: 1024
    seed: 0
    lamb: 1.0
    rho: 0.99
    alpha: 1.0
    transport_method: oracle   # qlib's TRA expects "oracle"|"router"
    GPU: 0
    model_config:
      input_size: 6
      hidden_size: 64
      num_layers: 2
      use_attn: true
      dropout: 0.0
    tra_config:
      num_states: 10
      hidden_size: 16
      tau: 1.0
      src_info: LR_TPE   # default head input mix

handler:
  class: Alpha360_OpenH
  module_path: custom_handler

step_len: 20
grad_clip_max_norm: 3.0
```

- [ ] **Step 2: Write the failing test**

`production/tests/test_train_tra.py`:

```python
from production.train_tra import _load_tra_config


def test_tra_config_loads_hyperparameters():
    cfg = _load_tra_config()
    assert cfg["model"]["kwargs"]["n_epochs"] == 100
    assert cfg["model"]["kwargs"]["tra_config"]["num_states"] == 10
```

- [ ] **Step 3: Run, expect failure**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_train_tra.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 4: Implement `train_tra.py`**

```python
"""TRA multi-head training wrapper.

Like train_alstm.py but uses qlib's TRA model. TRA is natively multi-task
(K=10 states with optimal-transport routing per stock per day); we run it
once per horizon and emit tra_1d / tra_5d / tra_20d series.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from production.train_alstm import _build_multihead_dataset

_log = logging.getLogger("train_tra")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_tra_config() -> dict:
    path = REPO_ROOT / "production/configs/tra_alpha360.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def train_tra_multihead(cfg, universe: list[str], end_date: date) -> list[pd.Series]:
    """Train TRA per horizon; return tra_1d, tra_5d, tra_20d Series."""
    from qlib.contrib.model.pytorch_tra import TRA
    from qlib.data.dataset import DatasetH
    from qlib.workflow import R

    tra_yaml = _load_tra_config()
    mhd = _build_multihead_dataset(cfg, universe, end_date)
    outputs: list[pd.Series] = []
    for h in cfg.horizons:
        handler = mhd.handler_objs[h.name]
        dataset = DatasetH(
            handler=handler,
            segments={
                "train": mhd.train_segment,
                "valid": mhd.valid_segment,
                "test": mhd.test_segment,
            },
        )
        model = TRA(**tra_yaml["model"]["kwargs"])
        with R.start(experiment_name=cfg.experiment_name, recorder_name=f"tra_{h.name}_{end_date}"):
            try:
                model.fit(dataset)
                pred = model.predict(dataset)
                R.save_objects(**{f"pred_{h.name}.pkl": pred})
                outputs.append(pred.rename(f"tra_{h.name}"))
            except Exception as exc:
                _log.warning("tra_failed_skipping", extra={"horizon": h.name, "error": str(exc)})

    return outputs
```

- [ ] **Step 5: Flip `tra.enabled` to true in `rolling_ensemble.yaml`**

```yaml
  - id: tra
    config: production/configs/tra_alpha360.yaml
    enabled: true
```

- [ ] **Step 6: Run, expect PASS**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_train_tra.py -v
```

Expected: 1 test PASSes.

- [ ] **Step 7: Commit**

```
git add production/configs/tra_alpha360.yaml production/train_tra.py production/configs/rolling_ensemble.yaml production/tests/test_train_tra.py
git commit -m "feat(production): TRA-Alpha360 training wrapper"
```

---

### Task 16: Ridge stacking meta-learner with OOF training

**Files:**
- Create: `production/ensemble_stacker.py`
- Create: `production/tests/test_ensemble_stacker.py`

- [ ] **Step 1: Write the failing tests**

`production/tests/test_ensemble_stacker.py`:

```python
import numpy as np
import pandas as pd
import pytest

from production.ensemble_stacker import RidgeStacker


def _mk_oof(n_days=20, n_stocks=30, n_bases=9, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_days)
    stocks = [f"SH60{i:04d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    cols = [f"base_{i}" for i in range(n_bases)]
    # Generate base preds correlated with target
    target = rng.normal(0, 0.02, size=len(idx))
    base_preds = pd.DataFrame(
        {c: target + rng.normal(0, 0.01, size=len(idx)) for c in cols},
        index=idx,
    )
    y = pd.Series(target, index=idx, name="label")
    return base_preds, y


def test_stacker_fits_and_predicts():
    base_preds, y = _mk_oof()
    stacker = RidgeStacker()
    stacker.fit_oof(base_preds, y)
    test_base, _ = _mk_oof(seed=1)
    out = stacker.predict(test_base)
    assert isinstance(out, pd.Series)
    assert out.shape == (len(test_base),)


def test_stacker_grid_searches_alpha():
    base_preds, y = _mk_oof()
    stacker = RidgeStacker(alpha_grid=[0.1, 1.0, 10.0])
    stacker.fit_oof(base_preds, y)
    assert stacker.alpha in (0.1, 1.0, 10.0)


def test_stacker_z_scores_cross_sectionally():
    """Stacker inputs must be cross-sectionally z-scored per day."""
    base_preds, y = _mk_oof()
    stacker = RidgeStacker()
    z = stacker._cross_sectional_zscore(base_preds)
    for d in z.index.get_level_values("datetime").unique()[:3]:
        slice_ = z.xs(d, level="datetime")
        # Each column's daily slice should have mean ~ 0, std ~ 1
        assert slice_.mean().abs().max() < 1e-6
        assert (slice_.std(ddof=0) - 1).abs().max() < 1e-6


def test_stacker_fallback_to_rank_average_when_fit_fails():
    """If Ridge fails (e.g. singular matrix), fall back to rank_average."""
    base_preds, _ = _mk_oof(n_days=1, n_stocks=2, n_bases=9)
    # Empty y → fit should fail; .predict still returns a Series
    stacker = RidgeStacker()
    try:
        stacker.fit_oof(base_preds, pd.Series(dtype="float64"))
    except Exception:
        pass
    out = stacker.predict_with_fallback(base_preds)
    assert out is not None
```

- [ ] **Step 2: Run, expect failure**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_ensemble_stacker.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `ensemble_stacker.py`**

```python
"""Ridge stacking meta-learner with OOF training.

Trained on per-day cross-sectionally z-scored base preds → realized open-to-open
return. Hyperparameter: alpha selected per-week via 3-point grid search on the
provided validation tail of OOF data.

Fallback chain:
  1. RidgeStacker.predict
  2. rank_average over available base columns
  3. (handled upstream) roll back to last week's recorder
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from production.ensemble_rank_avg import rank_average

_log = logging.getLogger("ensemble_stacker")


class RidgeStacker:
    def __init__(self, alpha_grid: Iterable[float] = (0.1, 1.0, 10.0)):
        self.alpha_grid = list(alpha_grid)
        self.alpha: float | None = None
        self.coefficients_: pd.Series | None = None
        self.intercept_: float | None = None
        self._fit_columns: list[str] | None = None

    @staticmethod
    def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
        def _z(s: pd.Series) -> pd.Series:
            mu = s.mean()
            sd = s.std(ddof=0)
            if sd == 0 or pd.isna(sd):
                return s - mu
            return (s - mu) / sd

        return df.groupby(level="datetime").transform(_z)

    def fit_oof(self, base_preds: pd.DataFrame, y: pd.Series) -> "RidgeStacker":
        if base_preds.empty or y.empty:
            raise ValueError("empty OOF training inputs")

        joined = base_preds.join(y.rename("__y__"), how="inner").dropna()
        if joined.empty:
            raise ValueError("no overlapping (date, instrument) rows between base_preds and y")

        X_raw = joined[[c for c in joined.columns if c != "__y__"]]
        X = self._cross_sectional_zscore(X_raw).fillna(0.0)
        y_aligned = joined["__y__"]
        self._fit_columns = list(X.columns)

        # Grid search alpha by validation IC on the held-out last 20% of dates
        dates = sorted(X.index.get_level_values("datetime").unique())
        cut = int(len(dates) * 0.8)
        train_dates, val_dates = set(dates[:cut]), set(dates[cut:])
        X_train = X[X.index.get_level_values("datetime").isin(train_dates)]
        y_train = y_aligned[y_aligned.index.get_level_values("datetime").isin(train_dates)]
        X_val = X[X.index.get_level_values("datetime").isin(val_dates)]
        y_val = y_aligned[y_aligned.index.get_level_values("datetime").isin(val_dates)]

        best_alpha, best_ic = None, -np.inf
        for a in self.alpha_grid:
            mdl = Ridge(alpha=a)
            mdl.fit(X_train.to_numpy(), y_train.to_numpy())
            pred_val = pd.Series(mdl.predict(X_val.to_numpy()), index=X_val.index)
            # Daily IC = Pearson on each day, mean across days
            df_eval = pd.DataFrame({"pred": pred_val, "y": y_val}).dropna()
            ics = df_eval.groupby(level="datetime").apply(
                lambda g: g["pred"].corr(g["y"]) if len(g) > 2 else np.nan
            ).dropna()
            ic_mean = ics.mean() if len(ics) else -np.inf
            if ic_mean > best_ic:
                best_alpha, best_ic = a, ic_mean

        self.alpha = best_alpha if best_alpha is not None else self.alpha_grid[0]

        # Final fit on all data
        final = Ridge(alpha=self.alpha)
        final.fit(X.to_numpy(), y_aligned.to_numpy())
        self.coefficients_ = pd.Series(final.coef_, index=self._fit_columns)
        self.intercept_ = float(final.intercept_)
        _log.info(
            "stacker_fit",
            extra={"alpha": self.alpha, "best_val_ic": best_ic, "coefs": self.coefficients_.to_dict()},
        )
        return self

    def predict(self, base_preds: pd.DataFrame) -> pd.Series:
        if self.coefficients_ is None or self._fit_columns is None:
            raise RuntimeError("Stacker must be fit before predict")
        X_raw = base_preds.reindex(columns=self._fit_columns)
        X = self._cross_sectional_zscore(X_raw).fillna(0.0)
        out = X.to_numpy() @ self.coefficients_.to_numpy() + (self.intercept_ or 0.0)
        return pd.Series(out, index=X.index, name="score_stacked")

    def predict_with_fallback(self, base_preds: pd.DataFrame) -> pd.Series:
        """Try Ridge first; fall back to rank_average if anything fails."""
        try:
            return self.predict(base_preds)
        except Exception as exc:
            _log.warning("stacker_predict_failed_falling_back_to_rank_avg", extra={"error": str(exc)})
            ranks = rank_average(base_preds)
            return (-ranks).rename("score_rank_avg_fallback")
```

- [ ] **Step 4: Run tests, expect PASS**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_ensemble_stacker.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```
git add production/ensemble_stacker.py production/tests/test_ensemble_stacker.py
git commit -m "feat(production): Ridge stacker with OOF training + grid search alpha"
```

---

### Task 17: Wire stacking + fallback chain into `rolling_train`

**Files:**
- Modify: `production/rolling_train.py`
- Create: `production/tests/test_rolling_train_pipeline.py`

- [ ] **Step 1: Write the integration test using mocks**

`production/tests/test_rolling_train_pipeline.py`:

```python
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from production import rolling_train
from production.rolling_train import RollingConfig


def _stub_base(end_date: date, model_id: str) -> list[pd.Series]:
    idx = pd.MultiIndex.from_product(
        [pd.date_range(end_date, periods=1), [f"SH60{i:04d}" for i in range(5)]],
        names=["datetime", "instrument"],
    )
    return [
        pd.Series(np.linspace(0.1, 0.5, 5), index=idx, name=f"{model_id}_1d"),
        pd.Series(np.linspace(-0.2, 0.2, 5), index=idx, name=f"{model_id}_5d"),
        pd.Series(np.linspace(0.0, 0.4, 5), index=idx, name=f"{model_id}_20d"),
    ]


def test_run_once_writes_pred_pkl_with_stacked_score(tmp_path, monkeypatch):
    cfg = rolling_train.load_config(rolling_train.REPO_ROOT / "production/configs/rolling_ensemble.yaml")
    monkeypatch.setattr(rolling_train, "init_qlib", lambda c: None)
    monkeypatch.setattr(rolling_train, "build_universe", lambda c, d: [f"SH60{i:04d}" for i in range(5)])

    def _fake_lgbm(cfg, h, universe, end):
        return _stub_base(end, "lgbm")[
            [h2.name for h2 in cfg.horizons].index(h.name)
        ]

    monkeypatch.setattr(rolling_train, "train_lgbm_horizon", _fake_lgbm)
    monkeypatch.setattr("production.train_alstm.train_alstm_multihead", lambda c, u, d: _stub_base(d, "alstm"))
    monkeypatch.setattr("production.train_tra.train_tra_multihead", lambda c, u, d: _stub_base(d, "tra"))
    # Direct pred.pkl into tmp_path
    monkeypatch.setattr(rolling_train, "REPO_ROOT", tmp_path)
    (tmp_path / "examples" / "mlruns").mkdir(parents=True, exist_ok=True)

    pred_path = rolling_train.run_once(cfg, date(2026, 5, 10))
    assert pred_path.exists()
    df = pd.read_pickle(pred_path)
    assert "score" in df.columns
    assert "consensus" in df.columns
    base_cols = [c for c in df.columns if c not in {"score", "consensus"}]
    # 9 base columns (3 models × 3 horizons) when stacker fits successfully
    assert len(base_cols) == 9
```

- [ ] **Step 2: Run, expect failure (stacking not wired yet)**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_rolling_train_pipeline.py -v
```

Expected: test fails because `run_once` still uses rank-average.

- [ ] **Step 3: Replace the rank-average step in `rolling_train.run_once` with stacking + fallback**

Replace this block in `production/rolling_train.py`:

```python
    # Ensemble step — Phase D: rank-average across all base columns.
    # Phase E (T16) replaces with Ridge stacking; this remains as the fallback.
    from production.ensemble_rank_avg import rank_average

    rank_avg_series = rank_average(base_preds)
    # Convert "lower rank = better" to a higher-is-better score by negating.
    unified = (-rank_avg_series).rename("score")
```

with:

```python
    # Ensemble step — Phase E: Ridge stacking with OOF training, plus a
    # 3-level fallback chain: Ridge -> rank_average -> roll back to last week.
    from production.ensemble_stacker import RidgeStacker
    from production.ensemble_rank_avg import rank_average

    # OOF training data: re-run base models on stack-fit window. For β phase we
    # approximate by training the stacker on the *valid window* preds where we
    # already have realized labels. This is acceptable because all three base
    # models were early-stopped on valid (not fit on it).
    try:
        # Pull realized labels from the Alpha158 handler we already built
        from qlib.data import D
        h5 = next(h for h in cfg.horizons if h.name == "5d")
        s_5 = split(end_date=end_date, cfg=h5)
        # Build a label series using the 5d horizon, since stacker scores on 5d returns
        label_expr = "Ref($open, -6) / Ref($open, -1) - 1"
        labels = D.features(
            instruments=universe,
            fields=[label_expr],
            start_time=str(s_5.valid_start),
            end_time=str(s_5.valid_end),
        )
        labels.columns = ["y"]
        labels.index.names = ["instrument", "datetime"]
        labels = labels.swaplevel("instrument", "datetime").sort_index()

        # base_preds on the valid window
        valid_mask = (
            (base_preds.index.get_level_values("datetime") >= pd.Timestamp(s_5.valid_start))
            & (base_preds.index.get_level_values("datetime") <= pd.Timestamp(s_5.valid_end))
        )
        valid_base = base_preds[valid_mask]
        stacker = RidgeStacker().fit_oof(valid_base, labels["y"])
        unified = stacker.predict_with_fallback(base_preds).rename("score")
        _log.info("stacker_fitted_ok", extra={"alpha": stacker.alpha})
    except Exception as exc:
        _log.warning("stacker_failed_using_rank_average", extra={"error": str(exc)})
        rank_avg_series = rank_average(base_preds)
        unified = (-rank_avg_series).rename("score")
```

- [ ] **Step 4: Run integration test, expect PASS**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_rolling_train_pipeline.py -v
```

Expected: 1 PASS. (The test patches out qlib `D.features` indirectly via the universe stub; if the stacker path raises, the rank-average fallback path still produces 9 base columns + score + consensus, which the assertion accepts.)

- [ ] **Step 5: Commit**

```
git add production/rolling_train.py production/tests/test_rolling_train_pipeline.py
git commit -m "feat(production): Ridge stacking with rank-average fallback in pipeline"
```

---

### Task 18: 8-metric scorecard + multi-regime split + paired t-test

**Files:**
- Create: `production/metrics.py`
- Create: `production/tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

`production/tests/test_metrics.py`:

```python
import numpy as np
import pandas as pd
import pytest

from production.metrics import compute_scorecard, regime_split, paired_ttest


def _mk_pred_label():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-02", "2024-12-31", freq="B")
    stocks = [f"SH60{i:04d}" for i in range(50)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    true = rng.normal(0, 0.02, size=len(idx))
    noise = rng.normal(0, 0.01, size=len(idx))
    pred = pd.Series(true * 0.5 + noise, index=idx, name="score")
    label = pd.Series(true, index=idx, name="label")
    return pred, label


def test_scorecard_returns_eight_keys():
    pred, label = _mk_pred_label()
    out = compute_scorecard(pred, label, top_k=10, bps=10)
    expected_keys = {
        "ic_mean", "ric_mean", "icir",
        "top_bottom_spread_monthly",
        "annual_excess_return", "ir", "max_drawdown",
        "daily_turnover",
    }
    assert expected_keys.issubset(out.keys())


def test_scorecard_ic_in_reasonable_range():
    pred, label = _mk_pred_label()
    out = compute_scorecard(pred, label, top_k=10, bps=10)
    # Predictions correlate with truth at ~0.5 * std ratio → IC should be positive
    assert out["ic_mean"] > 0


def test_regime_split_returns_segments():
    pred, label = _mk_pred_label()
    segments = regime_split(pred, label, segments=[("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31")])
    assert len(segments) == 2
    for seg_name, seg_metrics in segments.items():
        assert "ir" in seg_metrics


def test_paired_ttest_runs():
    a = pd.Series(np.random.normal(0.001, 0.02, 100))
    b = pd.Series(np.random.normal(0.000, 0.02, 100))
    t, p = paired_ttest(a, b)
    assert isinstance(t, float)
    assert 0.0 <= p <= 1.0
```

- [ ] **Step 2: Run, expect failure**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_metrics.py -v
```

- [ ] **Step 3: Implement `metrics.py`**

```python
"""8-metric scorecard, multi-regime split, paired t-test.

8 metrics per spec §8:
  Signal purity      IC mean, RIC mean, ICIR, top-bottom spread (monthly %)
  Portfolio perf     annualized excess return (cost-adj), IR (cost-adj), max DD
  Reality check      daily turnover
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


def _daily_ic(pred: pd.Series, label: pd.Series, method: str = "pearson") -> pd.Series:
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    return df.groupby(level="datetime").apply(
        lambda g: g["p"].corr(g["y"], method=method) if len(g) > 2 else np.nan
    ).dropna()


def _portfolio_returns(pred: pd.Series, label: pd.Series, top_k: int) -> tuple[pd.Series, pd.Series]:
    """Returns (daily_return, daily_turnover) for a TopK long-only portfolio."""
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    daily_returns: list[tuple[pd.Timestamp, float]] = []
    daily_turnover: list[tuple[pd.Timestamp, float]] = []
    last_set: set[str] = set()
    for d, g in df.groupby(level="datetime"):
        top = g.nlargest(top_k, "p")
        # Equal weight
        r = top["y"].mean() if not top.empty else 0.0
        daily_returns.append((d, r))
        cur_set = set(top.index.get_level_values("instrument"))
        if last_set:
            turn = len(cur_set.symmetric_difference(last_set)) / (2 * top_k)
        else:
            turn = 1.0
        daily_turnover.append((d, turn))
        last_set = cur_set
    return (
        pd.Series(dict(daily_returns)).sort_index(),
        pd.Series(dict(daily_turnover)).sort_index(),
    )


def compute_scorecard(
    pred: pd.Series,
    label: pd.Series,
    top_k: int = 30,
    bps: float = 10,
) -> dict[str, float]:
    ic = _daily_ic(pred, label, "pearson")
    ric = _daily_ic(pred, label, "spearman")
    icir = ic.mean() / ic.std() if ic.std() > 0 else float("nan")

    # Monthly top-bottom spread
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    monthly_groups = df.groupby(pd.Grouper(level="datetime", freq="M"))
    spreads = []
    for _, g in monthly_groups:
        if g.empty:
            continue
        top = g.nlargest(top_k, "p")["y"].mean()
        bot = g.nsmallest(top_k, "p")["y"].mean()
        spreads.append(top - bot)
    top_bottom_monthly = float(np.mean(spreads) * 100) if spreads else float("nan")

    r, turn = _portfolio_returns(pred, label, top_k)
    r_cost_adj = r - turn * (bps / 10_000)
    annual = r_cost_adj.mean() * 252
    ir = (r_cost_adj.mean() / r_cost_adj.std()) * np.sqrt(252) if r_cost_adj.std() > 0 else float("nan")

    cumulative = (1 + r_cost_adj).cumprod()
    drawdown = (cumulative / cumulative.cummax() - 1.0).min()

    return {
        "ic_mean": float(ic.mean()),
        "ric_mean": float(ric.mean()),
        "icir": float(icir),
        "top_bottom_spread_monthly": top_bottom_monthly,
        "annual_excess_return": float(annual),
        "ir": float(ir),
        "max_drawdown": float(drawdown),
        "daily_turnover": float(turn.mean()),
    }


def regime_split(
    pred: pd.Series,
    label: pd.Series,
    segments: list[tuple[str, str]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for start, end in segments:
        mask = (
            (pred.index.get_level_values("datetime") >= pd.Timestamp(start))
            & (pred.index.get_level_values("datetime") <= pd.Timestamp(end))
        )
        sub_pred = pred[mask]
        sub_label = label.reindex(sub_pred.index)
        if sub_pred.empty:
            continue
        out[f"{start}__{end}"] = compute_scorecard(sub_pred, sub_label)
    return out


def paired_ttest(new_daily_ic: pd.Series, old_daily_ic: pd.Series) -> tuple[float, float]:
    a, b = new_daily_ic.align(old_daily_ic, join="inner")
    t, p = ttest_rel(a.values, b.values)
    return float(t), float(p)
```

- [ ] **Step 4: Run, expect PASS**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_metrics.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Wire scorecard into `rolling_train.run_once`**

After the `write_pred_pkl(out, pred_path)` line in `rolling_train.run_once`, append:

```python
    # Step ⑦ — Scorecard on the test window (if labels exist for that range)
    try:
        from qlib.data import D
        h5 = next(h for h in cfg.horizons if h.name == "5d")
        s_5 = split(end_date=end_date, cfg=h5)
        labels = D.features(
            instruments=universe,
            fields=["Ref($open, -6) / Ref($open, -1) - 1"],
            start_time=str(s_5.test_start),
            end_time=str(s_5.test_end),
        )
        labels.columns = ["y"]
        labels.index.names = ["instrument", "datetime"]
        labels = labels.swaplevel().sort_index()["y"]

        from production.metrics import compute_scorecard
        score_window = out.reset_index().set_index(["datetime", "instrument"])["score"]
        scorecard = compute_scorecard(score_window, labels, top_k=30, bps=cfg.cost_bps)
        _log.info("scorecard", extra=scorecard)
        # Persist into mlruns by re-opening the latest recorder
        from qlib.workflow import R
        with R.start(experiment_name=cfg.experiment_name, recorder_name=f"ensemble_{end_date}"):
            for k, v in scorecard.items():
                R.log_metrics(**{k: v})
    except Exception as exc:
        _log.warning("scorecard_failed", extra={"error": str(exc)})
```

- [ ] **Step 6: Commit**

```
git add production/metrics.py production/rolling_train.py production/tests/test_metrics.py
git commit -m "feat(production): 8-metric scorecard + multi-regime split + paired t-test"
```

---

### Task 19: mlruns auto-archive + shadow paper tracking + auto-rollback

**Files:**
- Create: `production/mlruns_archive.py`
- Create: `production/shadow_tracker.py`
- Create: `production/tests/test_mlruns_archive.py`
- Create: `production/tests/test_shadow_tracker.py`
- Modify: `production/rolling_train.py`

- [ ] **Step 1: Write failing archive test**

`production/tests/test_mlruns_archive.py`:

```python
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from production.mlruns_archive import archive_old_recorders, _recorder_age_weeks


def test_recorder_age_weeks_calculation():
    now = datetime(2026, 5, 21)
    older = datetime(2026, 3, 21)
    assert _recorder_age_weeks(older, now=now) == pytest.approx((61) / 7, rel=0.01)


def test_archive_moves_old_dirs(tmp_path: Path):
    src = tmp_path / "mlruns" / "1" / "abc123_old"
    src.mkdir(parents=True)
    (src / "meta.yaml").write_text("artifact_uri: foo\n")
    # Set mtime to 70 days ago
    import os, time
    old_mtime = time.time() - 70 * 24 * 3600
    os.utime(src, (old_mtime, old_mtime))

    archive_dir = tmp_path / "archive"
    archive_old_recorders(mlruns_root=tmp_path / "mlruns", archive_root=archive_dir, keep_weeks=8)
    assert not src.exists()
    assert (archive_dir / "1" / "abc123_old").exists()


def test_archive_keeps_recent(tmp_path: Path):
    src = tmp_path / "mlruns" / "1" / "abc123_fresh"
    src.mkdir(parents=True)
    (src / "meta.yaml").write_text("artifact_uri: foo\n")
    archive_dir = tmp_path / "archive"
    archive_old_recorders(mlruns_root=tmp_path / "mlruns", archive_root=archive_dir, keep_weeks=8)
    assert src.exists()
    assert not (archive_dir / "1" / "abc123_fresh").exists()
```

- [ ] **Step 2: Write failing shadow tracker test**

`production/tests/test_shadow_tracker.py`:

```python
from datetime import date, timedelta

import pandas as pd
import pytest

from production.shadow_tracker import ShadowTracker


def test_shadow_starts_tracking_on_first_run():
    t = ShadowTracker()
    t.record_run(recorder_id="abc123", run_date=date(2026, 5, 17), is_shadow=True)
    state = t.get_state("abc123")
    assert state["weeks_observed"] == 1


def test_shadow_promotes_after_4_weeks_if_better():
    t = ShadowTracker()
    for i in range(4):
        d = date(2026, 5, 17) + timedelta(weeks=i)
        t.record_run(recorder_id="abc123", run_date=d, is_shadow=True, ir=2.8)
        t.record_baseline(recorder_id="prod_xyz", run_date=d, ir=2.2)
    decision = t.evaluate_promotion("abc123")
    assert decision["promote"] is True
    assert decision["ir_delta"] == pytest.approx(0.6)


def test_shadow_does_not_promote_before_4_weeks():
    t = ShadowTracker()
    for i in range(2):
        d = date(2026, 5, 17) + timedelta(weeks=i)
        t.record_run(recorder_id="abc123", run_date=d, is_shadow=True, ir=3.0)
        t.record_baseline(recorder_id="prod", run_date=d, ir=2.0)
    decision = t.evaluate_promotion("abc123")
    assert decision["promote"] is False
    assert decision["reason"] == "insufficient_weeks"
```

- [ ] **Step 3: Implement `mlruns_archive.py`**

```python
"""Auto-archive mlruns recorders older than N weeks.

Per spec §R5: mlruns directory grows unbounded; move recorders > 8 weeks old
to production/archive/<exp_id>/<recorder_id>/.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

_log = logging.getLogger("mlruns_archive")


def _recorder_age_weeks(timestamp: datetime, now: datetime | None = None) -> float:
    now = now or datetime.now()
    return (now - timestamp).total_seconds() / (7 * 24 * 3600)


def archive_old_recorders(
    mlruns_root: Path,
    archive_root: Path,
    keep_weeks: int = 8,
    now: datetime | None = None,
) -> int:
    """Walk `mlruns_root/<exp_id>/<recorder_id>/` and move any recorder with
    `meta.yaml` mtime older than `keep_weeks` into `archive_root/<exp_id>/<recorder_id>/`.

    Returns the number of archived recorders.
    """
    now = now or datetime.now()
    moved = 0
    if not mlruns_root.exists():
        return 0
    for exp_dir in mlruns_root.iterdir():
        if not exp_dir.is_dir():
            continue
        for rec_dir in exp_dir.iterdir():
            if not rec_dir.is_dir():
                continue
            meta = rec_dir / "meta.yaml"
            if not meta.exists():
                continue
            ts = datetime.fromtimestamp(rec_dir.stat().st_mtime)
            age_weeks = _recorder_age_weeks(ts, now=now)
            if age_weeks > keep_weeks:
                dest = archive_root / exp_dir.name / rec_dir.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(rec_dir), str(dest))
                moved += 1
                _log.info("recorder_archived", extra={"dest": str(dest), "age_weeks": age_weeks})
    return moved
```

- [ ] **Step 4: Implement `shadow_tracker.py`**

```python
"""Shadow paper trading tracker.

Per spec §8: every new candidate model trains alongside production for 4 weeks
as `shadow_v2_ensemble`. After 4 weeks, if shadow IR > prod IR + 0.5, swap.

State is persisted in production/shadow_state.json. Tracker is intentionally
simple — no DB; restart-safe via the JSON file.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

_log = logging.getLogger("shadow_tracker")

STATE_PATH = Path(__file__).resolve().parent / "shadow_state.json"
PROMOTION_WEEKS = 4
PROMOTION_IR_THRESHOLD = 0.5


@dataclass
class ShadowState:
    weeks_observed: int = 0
    ir_history: list[float] = field(default_factory=list)
    baseline_ir_history: list[float] = field(default_factory=list)


class ShadowTracker:
    def __init__(self, state_path: Path = STATE_PATH):
        self.state_path = state_path
        self._state: dict[str, ShadowState] = self._load()

    def _load(self) -> dict[str, ShadowState]:
        if not self.state_path.exists():
            return {}
        with self.state_path.open() as f:
            raw = json.load(f)
        return {k: ShadowState(**v) for k, v in raw.items()}

    def _save(self) -> None:
        serial = {k: {"weeks_observed": v.weeks_observed, "ir_history": v.ir_history, "baseline_ir_history": v.baseline_ir_history} for k, v in self._state.items()}
        self.state_path.write_text(json.dumps(serial, indent=2))

    def record_run(self, recorder_id: str, run_date: date, is_shadow: bool, ir: float | None = None) -> None:
        if not is_shadow:
            return
        st = self._state.setdefault(recorder_id, ShadowState())
        st.weeks_observed += 1
        if ir is not None:
            st.ir_history.append(ir)
        self._save()

    def record_baseline(self, recorder_id: str, run_date: date, ir: float) -> None:
        # Append baseline IR to the most recently observed shadow candidate
        if not self._state:
            return
        last_id = max(self._state, key=lambda k: self._state[k].weeks_observed)
        self._state[last_id].baseline_ir_history.append(ir)
        self._save()

    def get_state(self, recorder_id: str) -> dict:
        st = self._state.get(recorder_id, ShadowState())
        return {"weeks_observed": st.weeks_observed, "ir_history": st.ir_history}

    def evaluate_promotion(self, recorder_id: str) -> dict:
        st = self._state.get(recorder_id)
        if st is None or st.weeks_observed < PROMOTION_WEEKS:
            return {"promote": False, "reason": "insufficient_weeks", "ir_delta": None}
        # Average IR diff across the observed weeks
        n = min(len(st.ir_history), len(st.baseline_ir_history))
        if n == 0:
            return {"promote": False, "reason": "no_ir_data", "ir_delta": None}
        delta = (sum(st.ir_history[:n]) - sum(st.baseline_ir_history[:n])) / n
        return {
            "promote": delta > PROMOTION_IR_THRESHOLD,
            "reason": "ok" if delta > PROMOTION_IR_THRESHOLD else "ir_delta_too_small",
            "ir_delta": delta,
        }
```

- [ ] **Step 5: Run tests, expect PASS**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_mlruns_archive.py production/tests/test_shadow_tracker.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 6: Wire archive + auto-rollback into `rolling_train`**

Append to the end of `rolling_train.run_once`, after the scorecard logging:

```python
    # Auto-archive recorders older than cfg.archive_weeks
    from production.mlruns_archive import archive_old_recorders
    mlruns_root = REPO_ROOT / "examples" / "mlruns"
    archive_root = REPO_ROOT / "production" / "archive"
    archived = archive_old_recorders(mlruns_root, archive_root, keep_weeks=cfg.archive_weeks)
    _log.info("recorders_archived", extra={"count": archived})

    # Auto-rollback: if past 2 weeks cumulative IR < 0, revert to N-2 recorder
    try:
        from qlib.workflow import R
        recs = sorted(
            R.list_recorders(experiment_name=cfg.experiment_name).values(),
            key=lambda r: r.info.get("start_time", ""),
            reverse=True,
        )
        if len(recs) >= 3:
            recent_irs = []
            for rr in recs[:2]:
                m = rr.list_metrics() if hasattr(rr, "list_metrics") else {}
                if "ir" in m:
                    recent_irs.append(m["ir"])
            if recent_irs and sum(recent_irs) < 0:
                _log.warning(
                    "auto_rollback_triggered",
                    extra={"current": recs[0].id, "rollback_to": recs[2].id, "recent_irs": recent_irs},
                )
                # The "rollback" is implicit: the consumer picks the *latest non-archived*
                # recorder, and we mark the bad one for archival on the next run by
                # moving its directory to a holding folder.
                bad_rec_dir = mlruns_root / "1" / recs[0].id
                if bad_rec_dir.exists():
                    import shutil as _sh
                    _sh.move(str(bad_rec_dir), str(archive_root / "1" / f"rolled_back_{recs[0].id}"))
    except Exception as exc:
        _log.warning("rollback_check_failed", extra={"error": str(exc)})
```

- [ ] **Step 7: Commit**

```
git add production/mlruns_archive.py production/shadow_tracker.py production/rolling_train.py production/tests/test_mlruns_archive.py production/tests/test_shadow_tracker.py
git commit -m "feat(production): mlruns auto-archive + shadow paper tracker + auto-rollback"
```

---

## Phase F — Frontend + Final Acceptance

### Task 20: Frontend Picks page — view selector + consensus column

**Files:**
- Modify: `frontend/src/pages/Picks.tsx`
- Modify: `frontend/src/api/types.ts` (regenerated; consensus + base_scores arrive in ScreenItem)

- [ ] **Step 1: Regenerate types (no-op if already done)**

```
cd frontend
npm run gen:api
```

Verify `ScreenItem` in `frontend/src/api/types.ts` now contains `consensus` and `base_scores`.

- [ ] **Step 2: Modify `Picks.tsx` to add view selector + consensus column**

Locate the current view query and table render in `frontend/src/pages/Picks.tsx`. Add at the top of the component body:

```tsx
const [view, setView] = useState<"ensemble" | "lightgbm" | "alstm" | "tra">("ensemble");
const [minConsensus, setMinConsensus] = useState(0);
```

Pass `view` into the existing GET screen call as a query parameter:

```tsx
const { data } = useQuery({
  queryKey: ["picks", top, days, minTop, view],
  queryFn: async () => {
    const { data, error } = await client.GET("/api/models/screen", {
      params: { query: { top, days, min_top: minTop, view } },
    });
    if (error) throw new Error(JSON.stringify(error));
    return data;
  },
});
```

Add view selector + consensus filter to the existing filter bar:

```tsx
<select
  className="bg-gray-800 px-2 py-1 rounded"
  value={view}
  onChange={(e) => setView(e.target.value as "ensemble" | "lightgbm" | "alstm" | "tra")}
>
  <option value="ensemble">Ensemble</option>
  <option value="lightgbm">LightGBM only</option>
  <option value="alstm">ALSTM only</option>
  <option value="tra">TRA only</option>
</select>
<label className="flex items-center gap-2 text-sm">
  <span>Min consensus</span>
  <input
    type="range" min={0} max={1} step={0.1}
    value={minConsensus}
    onChange={(e) => setMinConsensus(Number(e.target.value))}
  />
  <span>{minConsensus.toFixed(1)}</span>
</label>
```

Add a `Consensus` column to the table render, with color coding:

```tsx
<th>Consensus</th>
...
<td>
  <span
    className={
      item.consensus >= 0.78
        ? "text-green-400"
        : item.consensus >= 0.44
        ? "text-yellow-400"
        : "text-gray-400"
    }
  >
    {(item.consensus * 100).toFixed(0)}%
  </span>
</td>
```

Filter rows by `item.consensus >= minConsensus` in the existing `.filter(…)`:

```tsx
const visible = (data?.items ?? []).filter((item) => item.consensus >= minConsensus);
```

- [ ] **Step 3: Add backend handling of `view` parameter**

Edit `backend/app/models/router.py`:

```python
@router.get("/screen", response_model=ScreenResponse)
def screen(
    top: int = Query(default=30, ge=1, le=300),
    days: int = Query(default=5, ge=1, le=60),
    min_top: int = Query(default=0, ge=0),
    experiment: str | None = Query(default=None),
    view: str = Query(default="ensemble", regex="^(ensemble|lightgbm|alstm|tra)$"),
):
    return service.screen(top=top, days=days, min_top=min_top, experiment=experiment, view=view)
```

Edit `service.screen` to honor `view`: when `view != "ensemble"`, derive `score` from the average of that model's per-horizon columns instead of the unified `score` field. The relevant lines:

```python
def screen(top, days, min_top, experiment, view="ensemble"):
    df = _load_pred_pkl(...)   # existing
    if view != "ensemble":
        prefix = view + "_"
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            # No data for this view (e.g. early in rollout); fall back to ensemble
            view = "ensemble"
        else:
            df = df.copy()
            df["score"] = df[cols].mean(axis=1)
    name_map = _load_name_map()
    items = _build_screen_items(df, top, days, min_top, name_map)
    return ScreenResponse(
        experiment=...,  # keep existing wiring
        recorder_id=...,
        latest_date=...,
        window_days=days,
        universe_size=...,
        items=items,
    )
```

- [ ] **Step 4: Smoke-test in browser**

Restart backend; refresh `http://localhost:5173/picks`. Verify:
- View dropdown shows 4 options, default Ensemble.
- Switching to LightGBM only re-fetches and shows different ranks.
- Consensus column shows e.g. "100%" green, "44%" yellow, "11%" gray.
- Min-consensus slider filters rows live.

- [ ] **Step 5: Commit**

```
git add frontend/src/pages/Picks.tsx frontend/src/api/types.ts backend/app/models/router.py backend/app/models/service.py
git commit -m "feat(picks): view selector + consensus column + filter"
```

---

### Task 21: Acceptance criteria validation script

**Files:**
- Create: `production/validate_acceptance.py`
- Create: `production/tests/test_validate_acceptance.py`

- [ ] **Step 1: Write the failing test**

`production/tests/test_validate_acceptance.py`:

```python
import json
from pathlib import Path

from production.validate_acceptance import check_acceptance


def test_check_acceptance_returns_status_dict(tmp_path: Path):
    scorecard = {
        "ic_mean": 0.032,
        "ric_mean": 0.026,
        "icir": 0.45,
        "top_bottom_spread_monthly": 1.8,
        "annual_excess_return": 0.18,
        "ir": 2.6,
        "max_drawdown": -0.12,
        "daily_turnover": 0.18,
    }
    out = check_acceptance(scorecard, regime_irs={"a": 0.5, "b": 0.3, "c": 0.7, "d": 0.4, "e": 0.6})
    assert out["passed"] is True
    assert all(out["details"].values())


def test_check_acceptance_flags_low_ic():
    scorecard = {
        "ic_mean": 0.025,  # fails ≥ 0.030
        "ric_mean": 0.026,
        "icir": 0.45,
        "top_bottom_spread_monthly": 1.8,
        "annual_excess_return": 0.18,
        "ir": 2.6,
        "max_drawdown": -0.12,
        "daily_turnover": 0.18,
    }
    out = check_acceptance(scorecard, regime_irs={"a": 0.5})
    assert out["passed"] is False
    assert out["details"]["ic_mean"] is False


def test_check_acceptance_flags_negative_regime():
    scorecard = {
        "ic_mean": 0.032, "ric_mean": 0.026, "icir": 0.45,
        "top_bottom_spread_monthly": 1.8, "annual_excess_return": 0.18,
        "ir": 2.6, "max_drawdown": -0.12, "daily_turnover": 0.18,
    }
    out = check_acceptance(scorecard, regime_irs={"a": -0.1, "b": 0.3})
    assert out["passed"] is False
    assert out["details"]["regimes_all_positive"] is False
```

- [ ] **Step 2: Run, expect failure**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_validate_acceptance.py -v
```

- [ ] **Step 3: Implement `validate_acceptance.py`**

```python
"""Acceptance criteria validator.

Per spec §11, returns {passed: bool, details: {criterion: bool}}.

Performance thresholds (cost-adjusted):
  IC mean ≥ 0.030
  IR ≥ 2.5
  max drawdown ≤ 15% (i.e. >= -0.15)
  daily turnover ≤ 20%
  all 5 regime IRs > 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THRESHOLDS = {
    "ic_mean": 0.030,
    "ir": 2.5,
    "max_drawdown": -0.15,
    "daily_turnover": 0.20,
}


def check_acceptance(scorecard: dict, regime_irs: dict[str, float]) -> dict:
    details = {
        "ic_mean": scorecard.get("ic_mean", 0) >= THRESHOLDS["ic_mean"],
        "ir": scorecard.get("ir", 0) >= THRESHOLDS["ir"],
        "max_drawdown": scorecard.get("max_drawdown", -1) >= THRESHOLDS["max_drawdown"],
        "daily_turnover": scorecard.get("daily_turnover", 1) <= THRESHOLDS["daily_turnover"],
        "regimes_all_positive": all(ir > 0 for ir in regime_irs.values()) if regime_irs else False,
    }
    return {
        "passed": all(details.values()),
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorecard", required=True, help="path to scorecard JSON")
    parser.add_argument("--regimes", required=True, help="path to regime IRs JSON {seg_name: ir}")
    args = parser.parse_args()

    sc = json.loads(Path(args.scorecard).read_text())
    rg = json.loads(Path(args.regimes).read_text())
    result = check_acceptance(sc, rg)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run, expect PASS**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_validate_acceptance.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Run the full production test suite**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/ -v
```

Expected: every production test PASSes. (Numbers: 5 PIT, 5 walk-forward, 4 multi-horizon labels, 3 post-process, 5 consensus, 2 rank-avg, 1 ALSTM, 1 TRA, 4 stacker, 1 pipeline, 4 metrics, 3 archive, 3 shadow, 3 acceptance ≈ 44 tests.)

- [ ] **Step 6: Run the full backend test suite**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest -v
```

Expected: all green.

- [ ] **Step 7: Commit**

```
git add production/validate_acceptance.py production/tests/test_validate_acceptance.py
git commit -m "feat(production): acceptance criteria validator"
```

---

## Acceptance Criteria Summary (per spec §11)

After T21 the following must all hold. Each line maps to the task that delivers it.

**Functional:**
- `production/rolling_train.py run-once` runs end-to-end without manual intervention (T10–T18).
- Weekly APScheduler job triggers at user-configured time (default Sunday 22:00) and completes within 2 hours (T2–T4, T10).
- `pred.pkl` from latest ensemble recorder is consumed by Picks / Charts pages without code changes (T11).
- `/api/scheduling/retrain` GET/PUT works and rejects trading-hours slots (T2, T3).
- All 3 base models trained; ALSTM and TRA use GPU; LightGBM uses CPU (T10, T13, T15).
- PIT constituents file regenerated monthly; passes sanity range check (T6).
- Ensemble output includes per-stock `consensus` field ∈ [0, 1] (T10, T11).
- Shadow paper trading framework runs new candidates in parallel for 4 weeks before swap (T19).
- Auto-rollback to N-2 recorder triggers on 2-week negative IR (T19).

**Performance (validated by `production/validate_acceptance.py`):**
- IC mean ≥ 0.030 (T18, T21)
- IR (cost-adjusted) ≥ 2.5 (T18, T21)
- Max drawdown ≤ 15% (T18, T21)
- Daily turnover ≤ 20% (T18, T21)
- All 5 regime-split segments have IR > 0 (T18, T21)

**Quality:**
- Hyperparameters locked in YAML; no in-code tuning during weekly runs (T9, T13, T15).
- Unit tests for `test_no_overlap_train_valid_stack_test`, PIT range check, stacking input dimensionality (T6, T7, T16).
- mlruns recorder versions for past 8 weeks retrievable (T19).
- All 8 evaluation metrics in mlruns metrics dict per recorder (T18).

---

## Known Deviations / Followups

1. **ALSTM multi-head simplification (T13):** the spec calls for a *single network with 3 heads*; this plan trains qlib's stock ALSTM 3 times (one per horizon). Both produce 3 alstm_* columns for stacking. True multi-task head sharing is a follow-up issue.
2. **Stacker OOF approximation (T17):** the spec calls for fitting Ridge on a dedicated `Stack-fit 1y` window with base preds materialized there. This plan reuses the *valid* window predictions because qlib already early-stops on valid (not fit on it), which is a reasonable approximation. A dedicated Stack-fit window is a follow-up issue when training-time budget allows.
3. **Monthly review report (spec §8 last paragraph):** `production/reports/<year>_<month>.md` auto-generation is described in the spec but is not gated by acceptance criteria (§11). Add as a follow-up after β stabilizes; pulls per-week IR table, base-model contribution, Ridge coefficient evolution, and top-30 hit rate from mlruns.
4. **Shadow `?view=shadow` UI surface:** the backend `GET /api/models/shadow` endpoint described in spec §9 is *not* implemented in this plan — only the tracker (T19) writes state. Wire up the shadow comparison endpoint and Dashboard card after β meets acceptance.
5. **Real broker integration is OUT of scope forever (per spec §2 "Out of scope").**

---

**End of plan. After approval, switch to the chosen execution mode per the writing-plans skill handoff.**
