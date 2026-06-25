from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.resources import PROFILES, apply_post_spawn, popen_creationflags, popen_env
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


JobCallable = Callable[..., Awaitable[None]]


def make_subprocess_retrain_job(python_path: str, repo_root: Path) -> JobCallable:
    """Return an async job that spawns `python -m production.rolling_train run-once`
    as a child process and writes its stdout to a per-job log file. Required
    because rolling_train blocks for many minutes of CPU/GPU work; running it
    inside the FastAPI event loop would freeze HTTP. The log file is tailed by
    the training layer for structured PROGRESS lines.
    """

    async def _job(job_id: str, log_path: Path, run_spec: list[str] | None = None, profile_name: str = "conservative") -> None:
        argv = run_spec if run_spec else ["run-once"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        prof = PROFILES.get(profile_name, PROFILES["conservative"])
        _log.info("retrain_subprocess_starting", job_id=job_id, argv=argv, python=python_path, cwd=str(repo_root), profile=profile_name)
        proc = await asyncio.create_subprocess_exec(
            python_path,
            "-m",
            "production.rolling_train",
            *argv,
            cwd=str(repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, **popen_env(prof)},
            creationflags=popen_creationflags(prof),
        )
        apply_post_spawn(proc.pid, prof)
        if proc.stdout is None:
            _log.error("retrain_subprocess_no_stdout", job_id=job_id)
            await proc.wait()
            raise RuntimeError("retrain subprocess produced no stdout pipe")
        with log_path.open("wb") as fh:
            async for line in proc.stdout:
                fh.write(line)
                fh.flush()
                _log.info("retrain_subprocess_stdout", line=line.decode(errors="replace").rstrip())
        rc = await proc.wait()
        if rc != 0:
            _log.error("retrain_subprocess_failed", job_id=job_id, returncode=rc)
            raise RuntimeError(f"rolling_train exited {rc}")
        _log.info("retrain_subprocess_ok", job_id=job_id)

    return _job


class SchedulerManager:
    """Wraps an AsyncIOScheduler and the single-row schedule config.

    Holds a single asyncio.Lock to guarantee that only one retrain runs at a
    time - whether triggered by cron, by Run-now, or by overlapping schedules.
    """

    JOB_ID = "retrain_weekly"
    NIGHTLY_JOB_ID = "nightly_inference"

    def __init__(self, job_fn: JobCallable, logs_dir: Path | None = None):
        self._scheduler = AsyncIOScheduler()
        self._raw_job_fn = job_fn
        self._running_lock = asyncio.Lock()
        # Lock binding to the event loop happens at first `await`. Construct this
        # inside the FastAPI lifespan to ensure the right loop is used.
        # Per-job subprocess stdout is written under logs_dir so the training
        # layer can tail structured PROGRESS lines. Defaults to <repo>/logs.
        self._logs_dir = logs_dir or (Path(__file__).resolve().parent.parent.parent.parent / "logs")
        self._started = False
        # Tracks manual run_now invocations that have been queued via add_job
        # but whose _gated_job_fn body has not yet started. Prevents a TOCTOU
        # race where two parallel run_now calls both pass the is_running check
        # before either has acquired the lock.
        self._pending_count = 0
        # Per-job tracking so the UI can recover progress after page
        # navigation. OrderedDict + FIFO eviction at MAX so a long-running
        # backend doesn't accumulate job entries forever (fix-1 audit
        # 2026-05-29 — small contributor to commit charge growth).
        # status ∈ {pending, running, done, failed}. kind ∈ {cron, manual}.
        from collections import OrderedDict as _OD
        self._MAX_JOBS = 50
        self._jobs: "_OD[str, dict]" = _OD()
        self._active_job_id: str | None = None

    def _log_path_for(self, job_id: str) -> Path:
        return self._logs_dir / f"api_retrain_{job_id}.log"

    def _remember_job(self, job_id: str, entry: dict) -> None:
        self._jobs[job_id] = entry
        self._jobs.move_to_end(job_id)
        while len(self._jobs) > self._MAX_JOBS:
            self._jobs.popitem(last=False)

    # === Job tracking API (read-only; used by HTTP routes) ===

    def get_job_status(self, job_id: str) -> dict | None:
        """Return per-job snapshot or None."""
        return self._jobs.get(job_id)

    def get_active_job(self) -> dict | None:
        """Return the most recent retrain job (running or finished). Lets
        the frontend recover progress after a page navigation. Returns
        None if no retrain has ever been launched this process."""
        if self._active_job_id and self._active_job_id in self._jobs:
            return self._jobs[self._active_job_id]
        if not self._jobs:
            return None
        return max(self._jobs.values(), key=lambda e: e.get("started_at") or "")

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

    async def run_now(
        self, session: AsyncSession, force: bool = False, run_spec: list[str] | None = None
    ) -> str:
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
            job_id = f"{self.JOB_ID}_manual_{now.timestamp():.0f}"
            # Pre-register so /jobs/active/peek can return it even before
            # _gated_job_fn flips it to "running".
            self._remember_job(job_id, {
                "job_id": job_id,
                "kind": "manual",
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "queued_at": datetime.now(tz=_CST).isoformat(),
                "error": None,
                "log_path": str(self._log_path_for(job_id)),
                "run_spec": run_spec,
            })
            self._active_job_id = job_id
            job = self._scheduler.add_job(
                self._gated_job_fn,
                trigger=None,
                id=job_id,
                kwargs={"_tracked_job_id": job_id},
            )
        except Exception:
            self._pending_count -= 1
            self._jobs.pop(job_id, None)
            self._active_job_id = None
            raise
        _log.info("run_now_scheduled", job_id=job.id)
        return job.id

    async def _gated_job_fn(self, _tracked_job_id: str | None = None) -> None:
        """Job wrapper that drops the call if another one is in-flight.

        Decrements _pending_count in `finally` so manual run_now invocations
        balance out. Cron-triggered fires do not increment the counter (the
        cron job is installed once and fires periodically), so the guarded
        decrement below is a no-op in that path.

        `_tracked_job_id` is set by run_now() so we can update the per-job
        status dict on transitions. Cron fires pass None → still create a
        new entry (kind=cron) on first fire so the UI sees scheduled runs
        with the same status surface as manual run-now jobs.
        """
        # If cron fired, mint an ad-hoc tracking entry now (the cron job_id
        # in APScheduler is reused across firings; we want one tracking
        # entry per firing).
        if _tracked_job_id is None:
            _tracked_job_id = f"{self.JOB_ID}_cron_{datetime.now(tz=_CST).timestamp():.0f}"
            self._remember_job(_tracked_job_id, {
                "job_id": _tracked_job_id,
                "kind": "cron",
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "queued_at": datetime.now(tz=_CST).isoformat(),
                "error": None,
                "log_path": str(self._log_path_for(_tracked_job_id)),
            })
            self._active_job_id = _tracked_job_id

        entry = self._jobs.get(_tracked_job_id)
        try:
            if self._running_lock.locked():
                _log.warning("retrain_job_skipped_already_running", job_id=_tracked_job_id)
                if entry is not None:
                    entry["status"] = "skipped"
                    entry["finished_at"] = datetime.now(tz=_CST).isoformat()
                return
            async with self._running_lock:
                if entry is not None:
                    entry["status"] = "running"
                    entry["started_at"] = datetime.now(tz=_CST).isoformat()
                await self._persist_run(job_id=_tracked_job_id, phase="start", entry=entry)
                try:
                    kind = (entry or {}).get("kind", "manual")
                    profile_name = "aggressive" if kind == "cron" else "conservative"
                    await self._raw_job_fn(
                        _tracked_job_id,
                        self._log_path_for(_tracked_job_id),
                        (entry or {}).get("run_spec"),
                        profile_name,
                    )
                    if entry is not None:
                        entry["status"] = "done"
                    await self._persist_run(job_id=_tracked_job_id, phase="done", entry=entry)
                except Exception as exc:
                    if entry is not None:
                        entry["status"] = "failed"
                        entry["error"] = str(exc)
                    await self._persist_run(job_id=_tracked_job_id, phase="failed", entry=entry)
                    _log.exception("retrain_subprocess_raised", job_id=_tracked_job_id)
                    raise
                finally:
                    if entry is not None:
                        entry["finished_at"] = datetime.now(tz=_CST).isoformat()
        finally:
            if self._pending_count > 0:
                self._pending_count -= 1

    def install_nightly_inference(self, *, enabled: bool, hour: int) -> None:
        """Install (or remove) a daily cron that runs daily_inference under the
        aggressive profile. Idempotent. Trading-hours guard applies at fire time."""
        try:
            self._scheduler.remove_job(self.NIGHTLY_JOB_ID)
        except JobLookupError:
            pass
        if not enabled:
            return
        self._scheduler.add_job(
            self._run_nightly_inference,
            trigger=CronTrigger(hour=hour, minute=0),
            id=self.NIGHTLY_JOB_ID,
            replace_existing=True,
        )

    async def _run_nightly_inference(self) -> None:
        from app.inference import service as inf
        now = datetime.now(tz=_CST)
        if is_trading_hours_cst(now):
            _log.warning("nightly_inference_skipped_trading_hours")
            return
        _log.info("nightly_inference_trigger")
        inf.trigger_inference(reason="nightly_scheduled", profile_name="aggressive")

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
                    rs = (entry or {}).get("run_spec")
                    if rs and "reblend" in rs and "--only" in rs:
                        scope, models = "single", [rs[rs.index("--only") + 1]]
                    else:
                        scope, models = "full", None
                    await store.record_run_start(
                        session, job_id=job_id,
                        kind=(entry or {}).get("kind", "manual"),
                        scope=scope, models=models, started_at=now,
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
