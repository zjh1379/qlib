"""Post-processing for ensemble scores.

- EWMA smoothing across consecutive trading days (cuts daily churn ~50%).
- Cost adjustment for backtest IR (turnover x bps -> return drag).
"""
from __future__ import annotations

import pandas as pd


def ewma_smooth(scores: pd.DataFrame, alpha: float = 0.5, score_col: str = "score") -> pd.DataFrame:
    """Apply per-instrument EWMA across the time index.

    Input: MultiIndex (datetime, instrument) with `score_col` column.
    Output: same shape, smoothed in-place on `score_col`.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")

    out = scores.copy()
    # groupby preserves ordering; .ewm(alpha=...).mean() handles the recursion
    out[score_col] = (
        out.groupby(level="instrument")[score_col]
        .transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    )
    return out


def cost_adjust(returns: pd.Series, turnover: pd.Series, bps: float = 10) -> pd.Series:
    """Subtract turnover x (bps / 10_000) from returns.

    Inputs must have the same index (typically per-day portfolio returns and
    per-day portfolio turnover in [0, 1]).
    """
    cost = turnover * (bps / 10_000)
    return returns - cost
