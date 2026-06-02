# production/backtest/regime.py
"""Market-regime exposure signal in [0,1]. Trailing-only (no lookahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_exposure(market_close: pd.Series, method: str = "trend_ma",
                     ma_window: int = 120, band: float = 0.10,
                     min_exposure: float = 0.0, vol_target: float | None = None,
                     vol_window: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Daily exposure in [0,1] from a synthetic market close.
    trend_ma: e = clip((close/MA - 1)/band + 0.5, min_exposure, 1); warm-up -> 1.0.
    Optional vol_target overlay: e *= min(1, vol_target/realized_vol); may push below floor.
    """
    if method != "trend_ma":
        raise ValueError(f"unknown method {method!r}")
    close = market_close.sort_index()
    ma = close.rolling(ma_window, min_periods=ma_window).mean()
    raw = (close / ma - 1.0) / band + 0.5
    e = raw.clip(lower=min_exposure, upper=1.0)
    e = e.where(ma.notna(), 1.0)  # warm-up: stay fully invested
    if vol_target is not None:
        ret = close.pct_change()
        rv = ret.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(periods_per_year)
        vt = (vol_target / rv).clip(upper=1.0)
        vt = vt.where(rv.notna(), 1.0)
        e = (e * vt).clip(lower=0.0, upper=1.0)  # vol can push to full risk-off
    return e.rename("exposure")
