"""Training job helpers: parse structured PROGRESS lines from a job's log file
and assemble a TrainingJobStatus from a SchedulerManager job entry.

Mirrors backend/app/data/service.py::_latest_progress / _tail_log (the proven
data-refresh progress mechanism). Kept self-contained to the training slice.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.training.schemas import TrainingJobStatus, TrainingProgress


def tail_log(log_path: Path, n_lines: int = 50) -> str | None:
    if not log_path.is_file():
        return None
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 64 * 1024)
            f.seek(size - read_size)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-n_lines:])
    except Exception:
        return None


def latest_progress(log_path: Path) -> TrainingProgress | None:
    """Return the latest parseable 'PROGRESS {json}' line, or None."""
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
            if not line.startswith("PROGRESS "):
                continue
            try:
                payload = json.loads(line[len("PROGRESS "):])
                return TrainingProgress(**payload)
            except Exception:
                continue
        return None
    except Exception:
        return None


def build_job_status(entry: dict) -> TrainingJobStatus:
    """Enrich a SchedulerManager job entry with parsed progress + log tail."""
    log_path_str = entry.get("log_path")
    progress = None
    log_tail = None
    if log_path_str:
        lp = Path(log_path_str)
        progress = latest_progress(lp)
        log_tail = tail_log(lp)
    return TrainingJobStatus(
        job_id=entry["job_id"],
        kind=entry.get("kind", "manual"),
        status=entry["status"],
        started_at=entry.get("started_at"),
        finished_at=entry.get("finished_at"),
        error=entry.get("error"),
        progress=progress,
        log_tail=log_tail,
    )
