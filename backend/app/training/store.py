"""training_runs persistence (async). The SchedulerManager job lifecycle writes
a row on start and updates it on finish; the history endpoint reads via list_runs."""
from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.training.orm import TrainingRunORM


async def record_run_start(
    session: AsyncSession, *, job_id: str, kind: str, scope: str,
    models: list[str] | None, started_at: str,
) -> None:
    """Insert (or update) a run row in 'running' state."""
    existing = await session.get(TrainingRunORM, job_id)
    if existing is None:
        session.add(TrainingRunORM(
            job_id=job_id, kind=kind, scope=scope,
            models_json=json.dumps(models) if models else None,
            status="running", started_at=started_at,
        ))
    else:
        existing.status = "running"
        existing.started_at = started_at
    await session.commit()


async def record_run_finish(
    session: AsyncSession, *, job_id: str, status: str,
    recorder_id: str | None, error: str | None, finished_at: str,
) -> None:
    """Update a run row's terminal state. No-op if the job_id is unknown."""
    row = await session.get(TrainingRunORM, job_id)
    if row is None:
        return
    row.status = status
    row.recorder_id = recorder_id
    row.error = error
    row.finished_at = finished_at
    await session.commit()


async def list_runs(session: AsyncSession, limit: int = 100) -> list[TrainingRunORM]:
    """Return run rows, newest first (by created_at)."""
    res = await session.execute(
        select(TrainingRunORM).order_by(TrainingRunORM.created_at.desc()).limit(limit)
    )
    return list(res.scalars().all())
