from fastapi import APIRouter, Query

from app.models import service
from app.models.schemas import (
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
    min_price: float | None = Query(default=None, ge=0, description="Inclusive lower bound on most-recent close (CNY/share)"),
    max_price: float | None = Query(default=None, ge=0, description="Inclusive upper bound on most-recent close (CNY/share)"),
):
    return service.screen(
        top=top, days=days, min_top=min_top, experiment=experiment, view=view,
        min_price=min_price, max_price=max_price,
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
