"""In-memory inference job tracking + subprocess launch.

Mirrors backend/app/scheduling/service.py + evaluation/service.py pattern.
Module-level dict holds job states; a lock prevents concurrent runs.

Spawns production/daily_inference.py as a subprocess so model loading
doesn't bloat the FastAPI process.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
import uuid
from datetime import date, datetime
from pathlib import Path

from app.inference.schemas import InferenceJob, InferenceStatus, TriggerResponse

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Module-level state (single-process backend assumption)
_JOBS: dict[str, InferenceJob] = {}
_ACTIVE_JOB_ID: str | None = None
_LOCK = threading.Lock()
_LAST_RUN_AT: str | None = None
_LAST_SUCCESS_AT: str | None = None
_LAST_ERROR: str | None = None

# Subprocess timeout — daily_inference should be well under this
SUBPROCESS_TIMEOUT_SECONDS = 600


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
        _JOBS[job_id] = InferenceJob(
            job_id=job_id,
            status="running",
            started_at=now,
            end_date=end_date.isoformat() if end_date else None,
            reason=reason,
        )
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

    log.info("inference_subprocess_start job_id=%s reason=%s cmd=%s",
             job_id, reason, " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT),
            capture_output=True, text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
        rc = proc.returncode
        log.info("inference_subprocess_end job_id=%s rc=%d", job_id, rc)

        # Best-effort parse "appended new_rows=N" from output
        new_rows = None
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        for line in combined.splitlines():
            if "appended new_rows=" in line:
                try:
                    new_rows = int(line.split("new_rows=")[1].split()[0])
                    break
                except Exception:
                    pass

        with _LOCK:
            job = _JOBS.get(job_id)
            if job:
                job.status = "done" if rc == 0 else "failed"
                job.finished_at = datetime.utcnow().isoformat()
                job.new_rows = new_rows
                if rc != 0:
                    job.error = (proc.stderr or "")[-2000:]
                    _LAST_ERROR = job.error
                else:
                    _LAST_SUCCESS_AT = job.finished_at
                    _LAST_ERROR = None
            _ACTIVE_JOB_ID = None

    except subprocess.TimeoutExpired:
        log.error("inference_subprocess_timeout job_id=%s", job_id)
        with _LOCK:
            job = _JOBS.get(job_id)
            if job:
                job.status = "failed"
                job.finished_at = datetime.utcnow().isoformat()
                job.error = f"subprocess timed out after {SUBPROCESS_TIMEOUT_SECONDS}s"
            _LAST_ERROR = "timeout"
            _ACTIVE_JOB_ID = None

    except Exception as exc:
        log.exception("inference_subprocess_error job_id=%s: %s", job_id, exc)
        with _LOCK:
            job = _JOBS.get(job_id)
            if job:
                job.status = "failed"
                job.finished_at = datetime.utcnow().isoformat()
                job.error = str(exc)[-2000:]
            _LAST_ERROR = str(exc)
            _ACTIVE_JOB_ID = None
