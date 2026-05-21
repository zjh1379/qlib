from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.scheduling.orm import RetrainScheduleORM
from app.scheduling.schemas import RetrainScheduleRead, RetrainScheduleUpdate

_log = get_logger("scheduling")

# CST trading hours (Shenzhen/Shanghai): the guard treats the full 09:30-15:00
# weekday window as "trading hours" to avoid mid-day model swap. Weekends are
# never trading hours.
_TRADING_OPEN = time(9, 30)
_TRADING_CLOSE = time(15, 0)

_CST = ZoneInfo("Asia/Shanghai")

# Anchor for weekday-sample arithmetic in update_schedule. 2026-01-05 is a
# Monday (weekday() == 0); adding `timedelta(days=day_of_week)` lands on the
# matching weekday inside the same week.
_WEEKDAY_ANCHOR = datetime(2026, 1, 5)


def is_trading_hours_cst(dt: datetime) -> bool:
    """Return True iff `dt` falls inside CST trading hours (Mon-Fri 09:30-15:00).

    `dt` is interpreted in CST: pass either a tz-aware datetime in
    Asia/Shanghai or a naive datetime that already represents CST clock time.
    Only `.weekday()` and `.time()` are inspected, so tz-awareness is
    irrelevant to the comparison itself.
    """
    if dt.weekday() >= 5:  # Sat=5, Sun=6
        return False
    t = dt.time()
    return _TRADING_OPEN <= t <= _TRADING_CLOSE


class TradingHoursViolation(Exception):
    pass


class AlreadyRunning(Exception):
    pass


JobCallable = Callable[[], Awaitable[None]]


def make_subprocess_retrain_job(python_path: str, repo_root: Path) -> JobCallable:
    """Return an async job that spawns `python -m production.rolling_train run-once`
    as a child process. Required because rolling_train blocks for 1.5-4 hours of
    CPU/GPU work; running it inside the FastAPI event loop would freeze HTTP.
    """

    async def _job() -> None:
        _log.info("retrain_subprocess_starting", python=python_path, cwd=str(repo_root))
        proc = await asyncio.create_subprocess_exec(
            python_path,
            "-m",
            "production.rolling_train",
            "run-once",
            cwd=str(repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        # Drain stdout so the buffer doesn't block the child
        if proc.stdout is None:
            _log.error("retrain_subprocess_no_stdout")
            await proc.wait()
            return
        async for line in proc.stdout:
            _log.info("retrain_subprocess_stdout", line=line.decode(errors="replace").rstrip())
        rc = await proc.wait()
        if rc != 0:
            _log.error("retrain_subprocess_failed", returncode=rc)
        else:
            _log.info("retrain_subprocess_ok")

    return _job


class SchedulerManager:
    """Wraps an AsyncIOScheduler and the single-row schedule config.

    Holds a single asyncio.Lock to guarantee that only one retrain runs at a
    time - whether triggered by cron, by Run-now, or by overlapping schedules.
    """

    JOB_ID = "retrain_weekly"

    def __init__(self, job_fn: JobCallable):
        self._scheduler = AsyncIOScheduler()
        self._raw_job_fn = job_fn
        self._running_lock = asyncio.Lock()
        # Lock binding to the event loop happens at first `await`. Construct this
        # inside the FastAPI lifespan to ensure the right loop is used.
        self._started = False
        # Tracks manual run_now invocations that have been queued via add_job
        # but whose _gated_job_fn body has not yet started. Prevents a TOCTOU
        # race where two parallel run_now calls both pass the is_running check
        # before either has acquired the lock.
        self._pending_count = 0

    async def start(self, session: AsyncSession) -> None:
        schedule = await self._read_row(session)
        self._scheduler.start()
        self._started = True
        if schedule.enabled:
            self._install_job(schedule)
        _log.info("scheduler_started", schedule=schedule.model_dump())

    async def stop(self) -> None:
        if self._started:
            # In-flight retrains run as subprocesses and outlive this shutdown.
            # Do not switch to wait=True - that would hang the FastAPI shutdown
            # for the full retrain duration (1.5-4 hours).
            self._scheduler.shutdown(wait=False)
            self._started = False
            _log.info("scheduler_stopped")

    @property
    def is_running(self) -> bool:
        return self._pending_count > 0 or self._running_lock.locked()

    def get_next_run_time(self) -> datetime | None:
        if not self._started:
            return None
        job = self._scheduler.get_job(self.JOB_ID)
        return job.next_run_time if job is not None else None

    async def get_schedule(self, session: AsyncSession) -> RetrainScheduleRead:
        sched = await self._read_row(session)
        # Always overlay the live APScheduler next-run-time on top of the
        # persisted row, in case the DB column wasn't updated this boot.
        nrt = self.get_next_run_time()
        if nrt is not None:
            sched = sched.model_copy(update={"next_run_at": nrt})
        return sched

    async def update_schedule(
        self, session: AsyncSession, payload: RetrainScheduleUpdate
    ) -> RetrainScheduleRead:
        # Reject trading-hours slots - sample any concrete weekday + the chosen
        # time to ask is_trading_hours_cst.
        weekday_sample = (
            _WEEKDAY_ANCHOR + timedelta(days=payload.day_of_week)
        ).replace(hour=payload.hour, minute=payload.minute)
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

        # Compute next-run-at from a transient trigger BEFORE committing so we
        # only need a single commit. The trigger here is not installed into the
        # scheduler; we install the real job below via _reinstall_job.
        if payload.enabled:
            trigger = CronTrigger(
                day_of_week=payload.day_of_week,
                hour=payload.hour,
                minute=payload.minute,
            )
            row.next_run_at = trigger.get_next_fire_time(None, datetime.now(tz=_CST))
        else:
            row.next_run_at = None

        await session.commit()
        await session.refresh(row)

        schedule = self._row_to_read(row)
        self._reinstall_job(schedule)
        _log.info("schedule_updated", schedule=schedule.model_dump())
        return schedule

    async def run_now(self, session: AsyncSession, force: bool = False) -> str:
        now = datetime.now(tz=_CST)
        if not force and is_trading_hours_cst(now):
            raise TradingHoursViolation(
                "run_now refused during trading hours; pass force=true to override"
            )
        if self.is_running:
            raise AlreadyRunning("a retrain is already running; wait for it to finish")
        # Increment BEFORE add_job so a concurrent run_now sees is_running=True
        # immediately (closes the TOCTOU window between add_job and the moment
        # _gated_job_fn starts executing).
        self._pending_count += 1
        try:
            job = self._scheduler.add_job(
                self._gated_job_fn,
                trigger=None,
                id=f"{self.JOB_ID}_manual_{now.timestamp():.0f}",
            )
        except Exception:
            self._pending_count -= 1
            raise
        _log.info("run_now_scheduled", job_id=job.id)
        return job.id

    async def _gated_job_fn(self) -> None:
        """Job wrapper that drops the call if another one is in-flight.

        Decrements _pending_count in `finally` so manual run_now invocations
        balance out. Cron-triggered fires do not increment the counter (the
        cron job is installed once and fires periodically), so the guarded
        decrement below is a no-op in that path.
        """
        try:
            if self._running_lock.locked():
                _log.warning("retrain_job_skipped_already_running")
                return
            async with self._running_lock:
                await self._raw_job_fn()
        finally:
            if self._pending_count > 0:
                self._pending_count -= 1

    def _install_job(self, schedule: RetrainScheduleRead) -> None:
        trigger = CronTrigger(
            day_of_week=schedule.day_of_week,
            hour=schedule.hour,
            minute=schedule.minute,
        )
        self._scheduler.add_job(
            self._gated_job_fn, trigger=trigger, id=self.JOB_ID, replace_existing=True
        )

    def _reinstall_job(self, schedule: RetrainScheduleRead) -> None:
        try:
            self._scheduler.remove_job(self.JOB_ID)
        except JobLookupError:
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
