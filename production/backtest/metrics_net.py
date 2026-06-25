"""Net-of-cost portfolio metrics computed from the engine's daily ledger."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Multi-regime segments (reused from production/metrics.py conventions).
DEFAULT_REGIMES: list[tuple[str, str]] = [
    ("2018-01-01", "2018-12-31"),   # bear
    ("2019-01-01", "2020-01-31"),   # recovery
    ("2020-02-01", "2021-02-28"),   # covid liquidity
    ("2021-03-01", "2022-12-31"),   # high vol
    ("2023-01-01", "2026-12-31"),   # AI rally / recent
]

_NAN_KEYS = ["net_cagr", "gross_cagr", "net_ir", "max_drawdown",
             "avg_turnover", "cost_drag_annual", "win_rate", "n_days"]


def net_metrics(daily: pd.DataFrame, periods_per_year: int = 252) -> dict:
    if daily is None or daily.empty or "net" not in daily:
        out = {k: float("nan") for k in _NAN_KEYS}
        out["n_days"] = 0
        return out
    r = daily["net"].dropna()
    if r.empty:
        out = {k: float("nan") for k in _NAN_KEYS}
        out["n_days"] = 0
        return out
    n = len(r)
    equity = (1 + r).cumprod()
    gross_equity = (1 + daily["gross"].reindex(r.index).fillna(0.0)).cumprod()

    def _cagr(eq: pd.Series) -> float:
        return float(eq.iloc[-1] ** (periods_per_year / n) - 1) if eq.iloc[-1] > 0 else float("nan")

    ir = float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else float("nan")
    dd = float((equity / equity.cummax() - 1).min())
    return {
        "net_cagr": _cagr(equity),
        "gross_cagr": _cagr(gross_equity),
        "net_ir": ir,
        "max_drawdown": dd,
        "avg_turnover": float(daily["turnover"].mean()),
        "cost_drag_annual": float(daily["cost"].mean() * periods_per_year),
        "win_rate": float((r > 0).mean()),
        "n_days": int(n),
    }


def net_regime(daily: pd.DataFrame, segments: list[tuple[str, str]] | None = None,
               periods_per_year: int = 252) -> dict[str, dict]:
    segments = segments or DEFAULT_REGIMES
    out: dict[str, dict] = {}
    idx = pd.to_datetime(daily.index)
    for start, end in segments:
        mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        sub = daily.loc[mask]
        if sub.empty:
            continue
        out[f"{start}__{end}"] = net_metrics(sub, periods_per_year)
    return out


def tail_stats(net) -> dict:
    """Left-tail / variance diagnostics for a per-period net-return Series — how
    brutal a concentrated (top-1/2) book is. `net` = engine daily ledger 'net' col
    (or any return Series)."""
    r = pd.Series(net).dropna()
    if r.empty:
        return {"ret_p05": float("nan"), "ret_p10": float("nan"),
                "ret_std": float("nan"), "neg_period_pct": float("nan"),
                "worst": float("nan"), "n": 0}
    return {
        "ret_p05": float(r.quantile(0.05)),
        "ret_p10": float(r.quantile(0.10)),
        "ret_std": float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
        "neg_period_pct": float((r < 0).mean()),
        "worst": float(r.min()),
        "n": int(len(r)),
    }


def period_metrics(returns, *, bars_per_period: int = 1, periods_per_year: int = 252) -> dict:
    """Annualized metrics for a PER-PERIOD net-return Series (one return per rebalance
    block of `bars_per_period` trading days). Distinct from net_metrics, which annualizes
    a per-DAY ledger. Keys match the exec_backtest / research-runner callers."""
    r = pd.Series(returns).dropna()
    n = len(r)
    if n == 0:
        return {"net_cagr": float("nan"), "calmar": float("nan"),
                "max_dd": float("nan"), "win": float("nan"), "n_periods": 0}
    eq = (1 + r).cumprod()
    last = float(eq.iloc[-1])
    cagr = (last ** (periods_per_year / (bars_per_period * n)) - 1) if last > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1).min())
    return {"net_cagr": cagr,
            "calmar": (cagr / abs(dd)) if abs(dd) > 1e-12 else float("nan"),
            "max_dd": dd, "win": float((r > 0).mean()), "n_periods": n}
