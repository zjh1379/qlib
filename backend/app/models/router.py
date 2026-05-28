from fastapi import APIRouter, Query

from app.core.exceptions import BusinessError
from app.models import service
from app.models.schemas import (
    CandidatesResponse,
    ExperimentsResponse,
    PredictionHistory,
    RollbackRequest,
    RollbackResponse,
    ScreenResponse,
    VersionResponse,
)

router = APIRouter()


@router.get("/screen", response_model=ScreenResponse)
def screen(
    top: int = Query(default=30, ge=1, le=300),
    days: int = Query(default=5, ge=1, le=60),
    min_top: int = Query(default=0, ge=0),
    experiment: str | None = Query(default=None),
    view: str = Query(default="ensemble", pattern="^(ensemble|lightgbm|alstm|tra)$"),
    # Existing price filter
    min_price: float | None = Query(default=None, ge=0, description="Inclusive lower bound on most-recent close (CNY/share)"),
    max_price: float | None = Query(default=None, ge=0, description="Inclusive upper bound on most-recent close (CNY/share)"),
    # Tier 1 filters (T4)
    pct_change_n: int = Query(default=5, description="Lookback in trading days for pct_change filter"),
    min_pct_change: float | None = Query(default=None, description="Min pct change over pct_change_n (e.g. 0.05 = +5%)"),
    max_pct_change: float | None = Query(default=None),
    min_amplitude: float | None = Query(default=None, ge=0),
    max_amplitude: float | None = Query(default=None, ge=0),
    min_vol_ratio: float | None = Query(default=None, ge=0),
    max_vol_ratio: float | None = Query(default=None, ge=0),
    new_high_n: int = Query(default=0, description="0=off, 20/60/120 = require close to be N-day high"),
    boards: str | None = Query(default=None, description="Comma list: main,gem,star,bj,etf"),
    exclude_st: bool = Query(default=True),
):
    if pct_change_n not in (1, 3, 5, 10, 20):
        raise BusinessError(
            detail="pct_change_n must be one of 1,3,5,10,20",
            code="bad_pct_change_n",
            context={"pct_change_n": pct_change_n},
        )
    if new_high_n not in (0, 20, 60, 120):
        raise BusinessError(
            detail="new_high_n must be one of 0,20,60,120",
            code="bad_new_high_n",
            context={"new_high_n": new_high_n},
        )
    return service.screen(
        top=top, days=days, min_top=min_top, experiment=experiment, view=view,
        min_price=min_price, max_price=max_price,
        pct_change_n=pct_change_n,
        min_pct_change=min_pct_change, max_pct_change=max_pct_change,
        min_amplitude=min_amplitude, max_amplitude=max_amplitude,
        min_vol_ratio=min_vol_ratio, max_vol_ratio=max_vol_ratio,
        new_high_n=new_high_n,
        boards=boards,
        exclude_st=exclude_st,
    )


@router.get("/predictions/{symbol}", response_model=PredictionHistory)
def predictions(
    symbol: str,
    days: int = Query(default=60, ge=1, le=365),
    experiment: str | None = Query(default=None),
    view: str = Query(default="ensemble", pattern="^(ensemble|lightgbm|alstm|tra)$"),
):
    return service.prediction_history(
        symbol=symbol, days=days, experiment=experiment, view=view
    )


@router.get("/experiments", response_model=ExperimentsResponse)
def experiments():
    return service.list_experiments()


@router.get("/version", response_model=VersionResponse)
def version():
    return service.version_info()


@router.post("/rollback", response_model=RollbackResponse)
def rollback(payload: RollbackRequest):
    return service.rollback_to(target=payload.target)


@router.get("/candidates", response_model=CandidatesResponse)
def candidates_endpoint(
    top: int = Query(default=300, ge=1, le=500),
    days: int = Query(default=5, ge=1, le=60),
    min_top: int = Query(default=0, ge=0),
    experiment: str | None = Query(default=None),
    view: str = Query(default="ensemble", pattern="^(ensemble|lightgbm|alstm|tra)$"),
    models: str | None = Query(
        default=None,
        description=(
            "Comma-separated list of base column names (e.g. "
            "'lgbm_1d,lgbm_5d,tra_5d') to use as the ensemble score. "
            "When set, takes precedence over `view`. Empty or 'all' → use "
            "the pool-time default score (e.g. v9 = 1d+5d cols)."
        ),
    ),
):
    """Return the full candidate pool (cached per recorder + view + models + base params).
    Frontend fetches this ONCE, then applies filter + sort client-side. No filter
    query params here — Tier 1 filters apply in the browser."""
    return service.candidates(
        top=top, days=days, min_top=min_top, experiment=experiment,
        view=view, models=models,
    )
