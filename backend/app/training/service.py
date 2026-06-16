"""Training job helpers: parse structured PROGRESS lines from a job's log file
and assemble a TrainingJobStatus from a SchedulerManager job entry.

Mirrors backend/app/data/service.py::_latest_progress / _tail_log (the proven
data-refresh progress mechanism). Kept self-contained to the training slice.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.service import list_recorders_with_summary
from app.training import store
from app.training.schemas import TrainingJobStatus, TrainingProgress, TrainingRunRow


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


async def _list_runs(session):
    return await store.list_runs(session)


def _is_candidate(rec) -> bool:
    if rec is None:
        return False
    exp = str(getattr(rec, "experiment", "") or "")
    name = str(getattr(rec, "run_name", "") or "")
    return exp.endswith("_candidates") or name.startswith("candidate_")


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
            experiment=getattr(rec, "experiment", None),
            is_candidate=_is_candidate(rec),
        ))
    for rec in recs:
        if rec.recorder_id in linked:
            continue
        rows.append(TrainingRunRow(
            status="historical", recorder_id=rec.recorder_id, run_name=rec.run_name,
            created_at=rec.created_at, ic_mean=rec.ic_mean, ir=rec.ir,
            acceptance_passed=rec.acceptance_passed,
            experiment=rec.experiment, is_candidate=_is_candidate(rec),
        ))
    rows.sort(key=lambda r: r.created_at or "", reverse=True)
    return rows
