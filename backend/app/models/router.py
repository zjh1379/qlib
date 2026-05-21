from fastapi import APIRouter, Query

from app.models import service
from app.models.schemas import (
    ExperimentsResponse,
    PredictionHistory,
    ScreenResponse,
)

router = APIRouter()


@router.get("/screen", response_model=ScreenResponse)
def screen(
    top: int = Query(default=30, ge=1, le=300),
    days: int = Query(default=5, ge=1, le=60),
    min_top: int = Query(default=0, ge=0),
    experiment: str | None = Query(default=None),
):
    return service.screen(top=top, days=days, min_top=min_top, experiment=experiment)


@router.get("/predictions/{symbol}", response_model=PredictionHistory)
def predictions(
    symbol: str,
    days: int = Query(default=60, ge=1, le=365),
    experiment: str | None = Query(default=None),
):
    return service.prediction_history(symbol=symbol, days=days, experiment=experiment)


@router.get("/experiments", response_model=ExperimentsResponse)
def experiments():
    return service.list_experiments()
