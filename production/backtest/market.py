# production/backtest/market.py
"""Synthetic broad-market proxy = equal-weight TRAILING daily return of the
universe, cumulated to a close series for the regime/trend signal."""
from __future__ import annotations

# Force installed qlib ahead of the uncompiled ./qlib source tree.
import sys as _sys
import sysconfig as _sysconfig
_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import pandas as pd

# Trailing 1-day return (Ref +1 = yesterday). NEVER use forward refs here.
MKT_RET_EXPR = "$close / Ref($close, 1) - 1"


def mean_market_return(ret: pd.Series) -> pd.Series:
    """Equal-weight cross-sectional mean return per datetime."""
    return ret.groupby(level="datetime").mean().sort_index().rename("mkt_ret")


def returns_to_close(mkt_ret: pd.Series) -> pd.Series:
    """Cumulative product -> synthetic market close (base 1.0)."""
    return (1.0 + mkt_ret.fillna(0.0)).cumprod().rename("market_close")


def load_market_proxy(instruments, start: str, end: str,
                      config_path: str = "production/configs/rolling_ensemble.yaml") -> pd.Series:
    """Load trailing 1d returns for `instruments`, equal-weight mean per day,
    cumulate to a synthetic market close Series indexed by datetime."""
    from production.qlib_features import load_series
    s = load_series(instruments, start, end, MKT_RET_EXPR, "mkt_ret", config_path=config_path)
    return returns_to_close(mean_market_return(s.dropna()))
