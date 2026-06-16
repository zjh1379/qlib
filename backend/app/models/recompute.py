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
from typing import Callable

from app.models.schemas import (
    RecomputeJob,
    RecomputeProgress,
    RecomputeTriggerResponse,
)

log = logging.getLogger(__name__)

# Must match frontend Picks WINDOW_DAYS / POOL_SIZE (see plan File Structure note).
CANDIDATES_WINDOW_K = 20
CANDIDATES_POOL_CAP = 300

# phase -> (lo, hi) overall-percent band. Tune after profiling.
_PHASE_BOUNDS: dict[str, tuple[int, int]] = {
    "load": (0, 15),
    "score": (15, 30),
    "metrics": (30, 90),
    "enrich": (90, 100),
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


def fetch_metrics_chunked(
    symbols: list[str],
    fetch_fn: Callable[[list[str]], dict],
    chunk_size: int = 50,
    emit: Callable[[int, int], None] | None = None,
) -> dict:
    """Call fetch_fn over `symbols` in batches, merging results and reporting
    per-batch progress via `emit(done, total)`. Splitting the one big
    D.features call into batches adds negligible overhead but gives the
    progress bar real, smooth movement during the dominant phase."""
    out: dict = {}
    total = len(symbols)
    for i in range(0, total, chunk_size):
        batch = symbols[i:i + chunk_size]
        out.update(fetch_fn(batch))
        done = min(i + chunk_size, total)
        if emit is not None:
            emit(done, total)
    return out
