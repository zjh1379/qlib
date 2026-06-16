# backend/app/models/recompute.py
"""Recompute job for the candidate pool — wraps service.candidates() in a
background thread that warms the lru_cache while reporting honest progress.

Progress is injected via a thread-local sink (NOT passed as a function arg,
which would break the lru_cache key on _candidates_cached). On a cache hit the
compute body never runs, so the job completes instantly with no progress — the
intended "already-computed = instant" behaviour.

Module-level imports MUST stay limited to schemas + stdlib. The import of
app.models.service happens lazily inside _run_recompute to avoid a circular
import (service imports the progress primitives from this module).
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from datetime import datetime

from app.models.schemas import (
    RecomputeJob,
    RecomputeProgress,
    RecomputeTriggerResponse,
)

log = logging.getLogger(__name__)

# Must match frontend Picks WINDOW_DAYS / POOL_SIZE (see plan File Structure note).
CANDIDATES_WINDOW_K = 20
CANDIDATES_POOL_CAP = 300

# phase -> (lo, hi) overall-percent band. Profiled 2026-06-16: metrics (one
# ~30s get_filter_metrics call) dominates wall-clock; load ~2s, score/enrich
# sub-second. So metrics owns most of the bar.
_PHASE_BOUNDS: dict[str, tuple[int, int]] = {
    "load": (0, 10),
    "score": (10, 15),
    "metrics": (15, 92),
    "enrich": (92, 100),
}

# Thread-local progress sink. Set by the recompute thread; unset (None) on
# normal GET threads -> emit_progress is a no-op there.
_progress_local = threading.local()


def phase_percent(phase: str, current: int, total: int) -> int:
    lo, hi = _PHASE_BOUNDS.get(phase, (0, 100))
    frac = (current / total) if total else 1.0
    frac = min(max(frac, 0.0), 1.0)
    return int(round(lo + (hi - lo) * frac))


def emit_progress(phase: str, current: int, total: int, message: str) -> None:
    """Report progress to the active recompute job, if any (else no-op)."""
    sink = getattr(_progress_local, "sink", None)
    if sink is None:
        return
    sink(RecomputeProgress(phase=phase, percent=phase_percent(phase, current, total),
                           message=message))


_MAX_JOBS = 20
_JOBS: "OrderedDict[str, RecomputeJob]" = OrderedDict()
_ACTIVE_ID: str | None = None
_LOCK = threading.Lock()


def get_job(job_id: str) -> RecomputeJob | None:
    return _JOBS.get(job_id)


def get_active_job() -> RecomputeJob | None:
    if _ACTIVE_ID and _ACTIVE_ID in _JOBS:
        return _JOBS[_ACTIVE_ID]
    return None


def trigger_recompute(view: str, models: list[str]) -> RecomputeTriggerResponse:
    global _ACTIVE_ID
    with _LOCK:
        if _ACTIVE_ID and _ACTIVE_ID in _JOBS and _JOBS[_ACTIVE_ID].status == "running":
            return RecomputeTriggerResponse(status="already_running", job_id=_ACTIVE_ID)
        job_id = uuid.uuid4().hex[:12]
        _JOBS[job_id] = RecomputeJob(
            job_id=job_id, status="running",
            started_at=datetime.utcnow().isoformat(),
            view=view, models=list(models),
            progress=RecomputeProgress(phase="load", percent=0, message="开始重算"),
        )
        _JOBS.move_to_end(job_id)
        while len(_JOBS) > _MAX_JOBS:
            old_id, _ = _JOBS.popitem(last=False)
            log.debug("evicted_old_recompute_job %s", old_id)
        _ACTIVE_ID = job_id

    threading.Thread(target=_run_recompute, args=(job_id, view, list(models)),
                     daemon=True).start()
    return RecomputeTriggerResponse(status="started", job_id=job_id)


def _run_recompute(job_id: str, view: str, models: list[str]) -> None:
    global _ACTIVE_ID
    from app.models import service  # lazy import: avoids circular import

    def sink(p: RecomputeProgress) -> None:
        with _LOCK:
            j = _JOBS.get(job_id)
            if j:
                j.progress = p

    _progress_local.sink = sink
    try:
        models_csv = ",".join(models) if models else None
        # Warm the lru_cache. View+models drive the heavy path; cache hit = instant.
        service.candidates(
            top=CANDIDATES_POOL_CAP, days=CANDIDATES_WINDOW_K, min_top=0,
            view=view, models=models_csv,
        )
        with _LOCK:
            j = _JOBS.get(job_id)
            if j:
                j.status = "done"
                j.finished_at = datetime.utcnow().isoformat()
                j.progress = RecomputeProgress(phase="done", percent=100, message="完成")
    except Exception as exc:  # noqa: BLE001 — record any failure on the job
        log.exception("recompute_failed job_id=%s: %s", job_id, exc)
        with _LOCK:
            j = _JOBS.get(job_id)
            if j:
                j.status = "failed"
                j.finished_at = datetime.utcnow().isoformat()
                j.error = str(exc)[:2000]
    finally:
        _progress_local.sink = None
        with _LOCK:
            _ACTIVE_ID = None
