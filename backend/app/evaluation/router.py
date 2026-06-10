from fastapi import APIRouter, HTTPException, Query

from app.evaluation import service
from app.evaluation.schemas import (
    CompareResult,
    EvalResult,
    EvalRunRequest,
    RecorderSummary,
)

router = APIRouter()


@router.get("/recorders", response_model=list[RecorderSummary])
def list_recorders():
    """Enumerate all qlib recorders across all experiments with lightweight
    summary info (pred date range + cached eval status). Cheap; no pred.pkl load."""
    return service.list_recorders_with_summary()


@router.get("/active/peek")
def active_peek():
    """Return list of currently-evaluating recorder_ids. Empty list when
    nothing is running. Frontend ActiveJobsBadge polls this so the chip
    persists across page navigation."""
    return service.get_active_evaluations()


@router.post("/run", response_model=EvalResult)
def run_evaluation(payload: EvalRunRequest):
    """Trigger evaluation for a recorder. Cached; pass force_refresh=true to recompute."""
    try:
        return service.evaluate_recorder(
            recorder_id=payload.recorder_id,
            top_k=payload.top_k,
            cost_bps=payload.cost_bps,
            force_refresh=payload.force_refresh,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/results/{recorder_id}", response_model=EvalResult)
def get_result(recorder_id: str):
    """Return the cached eval result for a recorder. 404 if never evaluated."""
    cached = service.get_cached_result(recorder_id)
    if cached is None:
        raise HTTPException(status_code=404, detail=f"recorder {recorder_id} has no cached eval")
    return cached


@router.get("/compare", response_model=CompareResult)
def compare(
    a: str = Query(..., description="recorder_id of baseline (A)"),
    b: str = Query(..., description="recorder_id of challenger (B)"),
    top_k: int = Query(default=30, ge=1, le=300),
    cost_bps: float = Query(default=10.0, ge=0),
):
    """Side-by-side compare of 2 recorders + paired t-test. Triggers eval on
    either or both if not yet cached."""
    try:
        return service.compare_recorders(a, b, top_k=top_k, cost_bps=cost_bps)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
