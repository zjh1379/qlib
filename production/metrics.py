"""8-metric scorecard, multi-regime split, paired t-test.

8 metrics per spec §8:
  Signal purity      IC mean, RIC mean, ICIR, top-bottom spread (monthly %)
  Portfolio perf     annualized excess return (cost-adj), IR (cost-adj), max DD
  Reality check      daily turnover
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


def _daily_ic(pred: pd.Series, label: pd.Series, method: str = "pearson") -> pd.Series:
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    return df.groupby(level="datetime").apply(
        lambda g: g["p"].corr(g["y"], method=method) if len(g) > 2 else np.nan
    ).dropna()


def _portfolio_returns(pred: pd.Series, label: pd.Series, top_k: int) -> tuple[pd.Series, pd.Series]:
    """Returns (daily_return, daily_turnover) for a TopK long-only portfolio."""
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    daily_returns: list[tuple[pd.Timestamp, float]] = []
    daily_turnover: list[tuple[pd.Timestamp, float]] = []
    last_set: set[str] = set()
    for d, g in df.groupby(level="datetime"):
        top = g.nlargest(top_k, "p")
        r = top["y"].mean() if not top.empty else 0.0
        daily_returns.append((d, r))
        cur_set = set(top.index.get_level_values("instrument"))
        if last_set:
            turn = len(cur_set.symmetric_difference(last_set)) / (2 * top_k)
        else:
            turn = 1.0
        daily_turnover.append((d, turn))
        last_set = cur_set
    return (
        pd.Series(dict(daily_returns)).sort_index(),
        pd.Series(dict(daily_turnover)).sort_index(),
    )


def compute_scorecard(
    pred: pd.Series,
    label: pd.Series,
    top_k: int = 30,
    bps: float = 10,
) -> dict[str, float]:
    ic = _daily_ic(pred, label, "pearson")
    ric = _daily_ic(pred, label, "spearman")
    icir = ic.mean() / ic.std() if ic.std() > 0 else float("nan")

    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    monthly_groups = df.groupby(pd.Grouper(level="datetime", freq="ME"))
    spreads = []
    for _, g in monthly_groups:
        if g.empty:
            continue
        top = g.nlargest(top_k, "p")["y"].mean()
        bot = g.nsmallest(top_k, "p")["y"].mean()
        spreads.append(top - bot)
    top_bottom_monthly = float(np.mean(spreads) * 100) if spreads else float("nan")

    r, turn = _portfolio_returns(pred, label, top_k)
    r_cost_adj = r - turn * (bps / 10_000)
    annual = r_cost_adj.mean() * 252
    ir = (r_cost_adj.mean() / r_cost_adj.std()) * np.sqrt(252) if r_cost_adj.std() > 0 else float("nan")

    cumulative = (1 + r_cost_adj).cumprod()
    drawdown = (cumulative / cumulative.cummax() - 1.0).min()

    return {
        "ic_mean": float(ic.mean()),
        "ric_mean": float(ric.mean()),
        "icir": float(icir),
        "top_bottom_spread_monthly": top_bottom_monthly,
        "annual_excess_return": float(annual),
        "ir": float(ir),
        "max_drawdown": float(drawdown),
        "daily_turnover": float(turn.mean()),
    }


def regime_split(
    pred: pd.Series,
    label: pd.Series,
    segments: list[tuple[str, str]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for start, end in segments:
        mask = (
            (pred.index.get_level_values("datetime") >= pd.Timestamp(start))
            & (pred.index.get_level_values("datetime") <= pd.Timestamp(end))
        )
        sub_pred = pred[mask]
        sub_label = label.reindex(sub_pred.index)
        if sub_pred.empty:
            continue
        out[f"{start}__{end}"] = compute_scorecard(sub_pred, sub_label)
    return out


def paired_ttest(new_daily_ic: pd.Series, old_daily_ic: pd.Series) -> tuple[float, float]:
    a, b = new_daily_ic.align(old_daily_ic, join="inner")
    t, p = ttest_rel(a.values, b.values)
    return float(t), float(p)
