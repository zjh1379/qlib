"""AI analysis REST endpoints.

Public:
  GET  /api/analysis/active/peek    -> AnalysisJob | null
  GET  /api/analysis/status         -> AnalysisStatus
  GET  /api/analysis/jobs/{job_id}  -> AnalysisJob
  GET  /api/analysis/{symbol}       -> AiAnalysis | null   (latest stored)
  POST /api/analysis/run-now        -> TriggerResponse

Localhost-only (called by daily_inference subprocess):
  POST /api/internal/analysis/refresh -> TriggerResponse
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis import service
from app.analysis.orm import AiAnalysisORM
from app.analysis.schemas import AiAnalysis, AnalysisJob, AnalysisStatus, TriggerResponse
from app.analysis.store import fetch_analyses
from app.core.db import get_session

router = APIRouter(prefix="/api/analysis", tags=["analysis"])
internal_router = APIRouter(prefix="/api/internal", tags=["internal"])


@router.get("/active/peek", response_model=AnalysisJob | None)
def active_peek():
    return service.get_active_job()


@router.get("/status", response_model=AnalysisStatus)
def status():
    return service.get_status()


@router.get("/jobs/{job_id}", response_model=AnalysisJob)
def get_job(job_id: str):
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(404, detail="job not found")
    return job


@router.post("/run-now", response_model=TriggerResponse)
def run_now():
    return service.trigger_analysis(reason="manual_ui")


@router.get("/{symbol}", response_model=AiAnalysis | None)
async def get_for_symbol(symbol: str, session: AsyncSession = Depends(get_session)):
    res = await session.execute(
        select(AiAnalysisORM).where(AiAnalysisORM.symbol == symbol,
                                    AiAnalysisORM.status != "failed")
        .order_by(AiAnalysisORM.as_of_date.desc()).limit(1)
    )
    row = res.scalars().first()
    if row is None:
        return None
    got = await fetch_analyses(session, [symbol], row.as_of_date)
    return got.get(symbol)


@internal_router.post("/analysis/refresh", response_model=TriggerResponse)
def internal_refresh(request: Request):
    client_host = request.client.host if request.client else None
    if client_host not in ("127.0.0.1", "localhost", "::1", "testclient"):
        raise HTTPException(403, detail="localhost only")
    return service.trigger_analysis(reason="data_refresh")
