from typing import Any

from pydantic import BaseModel, Field


class CandleBar(BaseModel):
    time: str = Field(..., description="ISO date YYYY-MM-DD")
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictionBar(BaseModel):
    """Synthetic candle representing model's predicted close for a target date.

    open = prior actual close
    close = prior actual close * (1 + score_at_T-2)
    high/low computed with a small spread for visibility
    score field is the raw model score for cross-reference
    """

    time: str
    open: float
    high: float
    low: float
    close: float
    score: float


class ChartPayload(BaseModel):
    symbol: str
    actual: list[CandleBar]
    predicted: list[PredictionBar]
    forecast: list[PredictionBar] = Field(
        default_factory=list,
        description="Future-only predicted bars (dates beyond last actual)",
    )
    meta: dict[str, Any] = Field(default_factory=dict)
