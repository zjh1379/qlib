"""Structured progress emitter for production training runs.

Prints exactly one line per call:

    PROGRESS {"phase":"train","current":3,"total":9,"message":"training lgbm"}

The backend captures the training subprocess's stdout into a per-job log file
and tails the latest PROGRESS line into the job status. This mirrors the
data-refresh progress mechanism (backend/app/data/service.py::_latest_progress
and production/incremental_refresh.py).

Printing is unconditional and has no side effects beyond stdout, so running a
training script directly from the CLI just shows these lines in the console.
"""
from __future__ import annotations

import json


def emit_progress(phase: str, current: int, total: int, message: str = "") -> None:
    """Emit one structured PROGRESS line to stdout (flushed)."""
    payload = {
        "phase": str(phase),
        "current": int(current),
        "total": int(total),
        "message": str(message),
    }
    # flush=True: the backend tails this promptly even though a retrain runs for
    # many minutes and Python would otherwise buffer stdout when not a tty.
    print("PROGRESS " + json.dumps(payload, ensure_ascii=False), flush=True)


def emit_recorder(recorder_id: str) -> None:
    """Emit the produced recorder id so the backend can link a training run to
    its recorder (parsed from the per-job log alongside PROGRESS lines)."""
    print(f"RECORDER {recorder_id}", flush=True)
