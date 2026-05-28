from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.exceptions import BusinessError
from app.scheduling.schemas import (
    RetrainScheduleRead,
    RetrainScheduleUpdate,
    RunNowResponse,
)
from app.scheduling.service import AlreadyRunning, SchedulerManager, TradingHoursViolation

router = APIRouter()

# Manager singleton populated in main.lifespan (T4).
_manager: SchedulerManager | None = None


def set_manager(m: SchedulerManager) -> None:
    global _manager
    _manager = m


def get_manager() -> SchedulerManager:
    if _manager is None:
        raise RuntimeError("SchedulerManager not initialized")
    return _manager


@router.get("/retrain", response_model=RetrainScheduleRead)
async def get_schedule(session: AsyncSession = Depends(get_session)):
    return await get_manager().get_schedule(session)


@router.put("/retrain", response_model=RetrainScheduleRead)
async def put_schedule(
    payload: RetrainScheduleUpdate,
    session: AsyncSession = Depends(get_session),
):
    try:
        return await get_manager().update_schedule(session, payload)
    except TradingHoursViolation as exc:
        # Project convention: BusinessError(detail, code=..., context=...) — the
        # global handler in main.py emits {"detail", "code", "context"} with
        # http_status from the class attribute (400 for BusinessError).
        raise BusinessError(
            str(exc),
            code="trigger_during_trading_hours",
        ) from exc


@router.post("/retrain/run-now", response_model=RunNowResponse)
async def run_now(
    force: bool = Query(default=False),
    session: AsyncSession = Depends(get_session),
):
    try:
        job_id = await get_manager().run_now(session, force=force)
        return RunNowResponse(status="started", job_id=job_id)
    except TradingHoursViolation as exc:
        return RunNowResponse(status="rejected", reason=str(exc))
    except AlreadyRunning as exc:
        return RunNowResponse(status="rejected", reason=str(exc))


@router.get("/retrain/jobs/{job_id}")
def retrain_job_status(job_id: str):
    """Return per-retrain-job snapshot (pending/running/done/failed)."""
    s = get_manager().get_job_status(job_id)
    if s is None:
        # 404-style — frontend treats as cleared/expired.
        return None
    return s


@router.get("/retrain/jobs/active/peek")
def retrain_active_peek():
    """Return the most recent retrain job (running or finished). Used by
    the global ActiveJobsBadge so progress survives page navigation."""
    return get_manager().get_active_job()
