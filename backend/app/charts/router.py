from fastapi import APIRouter, Query

from app.charts.schemas import ChartPayload
from app.charts.service import get_chart

router = APIRouter()


@router.get("/{symbol}", response_model=ChartPayload)
async def chart(
    symbol: str,
    start: str = Query(..., description="ISO date YYYY-MM-DD"),
    end: str = Query(..., description="ISO date YYYY-MM-DD"),
    with_pred: bool = Query(default=True),
    experiment: str | None = Query(default=None),
) -> ChartPayload:
    return get_chart(symbol=symbol, start=start, end=end, with_pred=with_pred, experiment=experiment)
