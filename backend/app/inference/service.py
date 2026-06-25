"""In-memory inference job tracking + subprocess launch.

Mirrors backend/app/scheduling/service.py + evaluation/service.py pattern.
Module-level dict holds job states; a lock prevents concurrent runs.

Spawns production/daily_inference.py as a subprocess so model loading
doesn't bloat the FastAPI process.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import uuid
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path

from app.core.resources import PROFILES, apply_post_spawn, popen_creationflags, popen_env
from app.inference.schemas import InferenceJob, InferenceStatus, TriggerResponse

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Module-level state (single-process backend assumption).
# OrderedDict + MAX bound so long-running backends don't accumulate job
# entries forever (cause of slow memory growth -> contributes to commit
# charge approaching pagefile limit -> Windows freezes).
_MAX_JOBS = 50
_JOBS: "OrderedDict[str, InferenceJob]" = OrderedDict()
_ACTIVE_JOB_ID: str | None = None
_LOCK = threading.Lock()
_LAST_RUN_AT: str | None = None
_LAST_SUCCESS_AT: str | None = None
_LAST_ERROR: str | None = None

# Subprocess timeout — daily_inference should be well under this
SUBPROCESS_TIMEOUT_SECONDS = 600


def _remember_job(job_id: str, job: InferenceJob) -> None:
    """Insert a job into _JOBS with FIFO eviction at _MAX_JOBS."""
    _JOBS[job_id] = job
    _JOBS.move_to_end(job_id)
    while len(_JOBS) > _MAX_JOBS:
        old_id, _ = _JOBS.popitem(last=False)
        log.debug("evicted_old_inference_job %s", old_id)


def get_active_job() -> InferenceJob | None:
    if _ACTIVE_JOB_ID and _ACTIVE_JOB_ID in _JOBS:
        return _JOBS[_ACTIVE_JOB_ID]
    return None


def get_status() -> InferenceStatus:
    return InferenceStatus(
        last_run_at=_LAST_RUN_AT,
        last_success_at=_LAST_SUCCESS_AT,
        last_error=_LAST_ERROR,
        is_running=_ACTIVE_JOB_ID is not None,
    )


def get_job(job_id: str) -> InferenceJob | None:
    return _JOBS.get(job_id)


def trigger_inference(
    force: bool = False,
    end_date: date | None = None,
    reason: str = "manual",
) -> TriggerResponse:
    """Start daily_inference subprocess if not already running."""
    global _ACTIVE_JOB_ID, _LAST_RUN_AT

    with _LOCK:
        if _ACTIVE_JOB_ID and _ACTIVE_JOB_ID in _JOBS \
           and _JOBS[_ACTIVE_JOB_ID].status == "running":
            return TriggerResponse(status="already_running",
                                    job_id=_ACTIVE_JOB_ID)

        job_id = uuid.uuid4().hex[:12]
        now = datetime.utcnow().isoformat()
        _remember_job(job_id, InferenceJob(
            job_id=job_id,
            status="running",
            started_at=now,
            end_date=end_date.isoformat() if end_date else None,
            reason=reason,
        ))
        _ACTIVE_JOB_ID = job_id
        _LAST_RUN_AT = now

    # Spawn outside the lock
    thread = threading.Thread(
        target=_run_subprocess,
        args=(job_id, end_date, force, reason),
        daemon=True,
    )
    thread.start()
    return TriggerResponse(status="started", job_id=job_id)


def _run_subprocess(job_id: str, end_date: date | None, force: bool, reason: str):
    global _ACTIVE_JOB_ID, _LAST_SUCCESS_AT, _LAST_ERROR

    cmd = [sys.executable, "-m", "production.daily_inference"]
    if end_date:
        cmd += ["--end-date", end_date.isoformat()]
    if force:
        cmd += ["--force"]

    # Stream stdout/stderr to a per-job logfile rather than capture_output=True.
    # capture_output buffers ALL output in this process's memory until the
    # subprocess exits — verbose qlib/mlflow output (hundreds of lines per
    # run) at megabytes per run × repeated invocations causes the backend
    # RSS to grow unboundedly and was a contributor to the Windows commit
    # charge exhaustion (see Event 2004 audit 2026-05-29).
    log_path = LOG_DIR / f"inference_{job_id}.log"
    log.info("inference_subprocess_start job_id=%s reason=%s log=%s cmd=%s",
             job_id, reason, log_path, " ".join(cmd))
    new_rows: int | None = None
    rc: int | None = None
    err_tail: str = ""
    try:
        profile = PROFILES["conservative"]  # manual/UI inference keeps desktop responsive
        with log_path.open("wb") as logf:
            proc = subprocess.Popen(
                cmd, cwd=str(REPO_ROOT),
                stdout=logf, stderr=subprocess.STDOUT,
                env={**os.environ, **popen_env(profile)},
                creationflags=popen_creationflags(profile),
            )
            apply_post_spawn(proc.pid, profile)
            try:
                rc = proc.wait(timeout=SUBPROCESS_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                rc = -1
                err_tail = f"subprocess timed out after {SUBPROCESS_TIMEOUT_SECONDS}s"
                log.error("inference_subprocess_timeout job_id=%s", job_id)
        log.info("inference_subprocess_end job_id=%s rc=%d log=%s",
                 job_id, rc, log_path)
        # Read just the tail of the logfile for parse + error display (small)
        try:
            sz = log_path.stat().st_size
            with log_path.open("rb") as f:
                if sz > 4096:
                    f.seek(sz - 4096)
                tail = f.read().decode("utf-8", errors="replace")
            for line in tail.splitlines():
                if "appended new_rows=" in line:
                    try:
                        new_rows = int(line.split("new_rows=")[1].split()[0])
                    except Exception:
                        pass
            if rc != 0 and not err_tail:
                err_tail = tail[-2000:]
        except Exception:
            pass
    except Exception as exc:
        log.exception("inference_subprocess_error job_id=%s: %s", job_id, exc)
        err_tail = str(exc)[-2000:]
        rc = rc if rc is not None else -2
    finally:
        with _LOCK:
            job = _JOBS.get(job_id)
            if job:
                job.status = "done" if rc == 0 else "failed"
                job.finished_at = datetime.utcnow().isoformat()
                job.new_rows = new_rows
                if rc != 0:
                    job.error = err_tail
                    _LAST_ERROR = err_tail
                else:
                    _LAST_SUCCESS_AT = job.finished_at
                    _LAST_ERROR = None
            _ACTIVE_JOB_ID = None
