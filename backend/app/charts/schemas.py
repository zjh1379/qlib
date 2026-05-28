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


class HorizonMarker(BaseModel):
    """Per-horizon future prediction packet for a single symbol's chart.

    target_price = last_actual_close * (1 + pred_return) when pred_return is
    not None; otherwise equal to last_actual_close (rendered as flat marker).
    """
    horizon: str  # "1d" | "5d" | "20d"
    target_date: str
    target_price: float
    pred_return: float | None = None
    percentile: float
    model_agreement: float | None = None
    raw_scores: dict[str, float] = Field(default_factory=dict)


class ChartPayload(BaseModel):
    symbol: str
    actual: list[CandleBar]
    predicted: list[PredictionBar]
    forecast: list[PredictionBar] = Field(
        default_factory=list,
        description="Future-only predicted bars (dates beyond last actual)",
    )
    horizon_markers: list[HorizonMarker] = Field(
        default_factory=list,
        description="Future markers for 1d/5d/20d horizons drawn on the K-line",
    )
    meta: dict[str, Any] = Field(default_factory=dict)
