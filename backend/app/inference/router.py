"""Inference REST endpoints.

Public:
  GET  /api/inference/active/peek   -> InferenceJob | null
  GET  /api/inference/status        -> InferenceStatus
  GET  /api/inference/jobs/{job_id} -> InferenceJob
  POST /api/inference/run-now       -> TriggerResponse

Localhost-only (called by daily_inference subprocess):
  POST /api/internal/cache/invalidate -> {cleared: int}
"""
from fastapi import APIRouter, HTTPException, Request

from app.inference import service
from app.inference.schemas import InferenceJob, InferenceStatus, TriggerResponse

router = APIRouter(prefix="/api/inference", tags=["inference"])
internal_router = APIRouter(prefix="/api/internal", tags=["internal"])


@router.get("/active/peek", response_model=InferenceJob | None)
def active_peek():
    return service.get_active_job()


@router.get("/status", response_model=InferenceStatus)
def inference_status():
    return service.get_status()


@router.get("/jobs/{job_id}", response_model=InferenceJob)
def get_job(job_id: str):
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(404, detail="job not found")
    return job


@router.post("/run-now", response_model=TriggerResponse)
def trigger(force: bool = False):
    return service.trigger_inference(force=force, reason="manual_ui")


@internal_router.post("/cache/invalidate")
def invalidate(request: Request):
    """Called by daily_inference subprocess after appending to recorder.
    Restricted to localhost so external traffic can't flush the cache."""
    client_host = request.client.host if request.client else None
    if client_host not in ("127.0.0.1", "localhost", "::1", "testclient"):
        raise HTTPException(403, detail="localhost only")
    from app.models.service import invalidate_candidates_cache
    cleared = invalidate_candidates_cache()
    return {"cleared": cleared}
