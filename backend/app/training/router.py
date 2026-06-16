from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.scheduling.router import get_manager
from app.scheduling.service import AlreadyRunning, TradingHoursViolation
from app.training.schemas import TrainingJobStatus, TrainingRunRow, TrainRequest

router = APIRouter()


@router.post("/run")
async def run_training(
    payload: TrainRequest,
    session: AsyncSession = Depends(get_session),
):
    """Start a retrain. Reuses the shared SchedulerManager so the concurrency
    lock + trading-hours guard are shared with cron/run-now. scope="single"
    retrains a single algo via `reblend --only <model>`; scope="full" runs the
    full ensemble pipeline (run_spec=None)."""
    run_spec = None
    if payload.scope == "single":
        if len(payload.models) != 1:
            from app.core.exceptions import BusinessError
            raise BusinessError("scope=single requires exactly one model", code="bad_single_models")
        run_spec = ["reblend", "--only", payload.models[0]]
    try:
        job_id = await get_manager().run_now(session, force=payload.force, run_spec=run_spec)
        return {"status": "started", "job_id": job_id}
    except TradingHoursViolation as exc:
        return {"status": "rejected", "reason": str(exc)}
    except AlreadyRunning as exc:
        return {"status": "rejected", "reason": str(exc)}


@router.get("/jobs/active", response_model=TrainingJobStatus | None)
def active_job():
    from app.training.service import build_job_status
    entry = get_manager().get_active_job()
    return build_job_status(entry) if entry is not None else None


@router.get("/jobs/{job_id}", response_model=TrainingJobStatus | None)
def job_status(job_id: str):
    from app.training.service import build_job_status
    entry = get_manager().get_job_status(job_id)
    return build_job_status(entry) if entry is not None else None


@router.get("/runs", response_model=list[TrainingRunRow])
async def training_runs(session: AsyncSession = Depends(get_session)):
    from app.training.service import build_history
    return await build_history(session)
