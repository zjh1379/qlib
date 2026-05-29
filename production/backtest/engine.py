# production/backtest/engine.py
"""Holding-period-aware portfolio simulator.

Decouples the model SCORE (ranking only) from realized 1-day open->open
returns (P&L). Costs charged in yuan on actual traded notional per order.
"""
from __future__ import annotations

import pandas as pd

from .costs import CostModel
from .rebalance import RebalancePolicy


def run_backtest(
    scores,
    fwd_ret: pd.Series,
    policy: RebalancePolicy,
    cost: CostModel,
    capital: float = 100_000.0,
    score_col: str = "score",
) -> dict:
    """
    scores: Series (datetime,instrument)->score, OR DataFrame with `score_col`.
    fwd_ret: Series (datetime,instrument)->open(d+2)/open(d+1)-1.
    Returns dict: {"daily": DataFrame[gross,cost,net,turnover,nav], "final_nav", "capital"}.
    """
    s = scores[score_col] if isinstance(scores, pd.DataFrame) else scores
    fwd_dates = set(fwd_ret.index.get_level_values("datetime").unique())
    dates = [d for d in sorted(s.index.get_level_values("datetime").unique()) if d in fwd_dates]

    current = pd.Series(dtype=float)
    nav = float(capital)
    rows = []
    for i, d in enumerate(dates):
        s_d = s.xs(d, level="datetime")
        cost_frac = 0.0
        turnover = 0.0
        if policy.should_rebalance(i):
            target = policy.target_weights(s_d, current)
            allidx = current.index.union(target.index)
            cur = current.reindex(allidx).fillna(0.0)
            tgt = target.reindex(allidx).fillna(0.0)
            delta = tgt - cur
            turnover = float(0.5 * delta.abs().sum())
            total_cost = 0.0
            for inst, dw in delta.items():
                if dw == 0:
                    continue
                total_cost += cost.trade_cost(abs(dw) * nav, is_buy=dw > 0)
            cost_frac = total_cost / nav if nav > 0 else 0.0
            current = target
        r_d = fwd_ret.xs(d, level="datetime")
        gross = float((current * r_d.reindex(current.index).fillna(0.0)).sum())
        net = gross - cost_frac
        nav *= (1 + net)
        rows.append({"datetime": d, "gross": gross, "cost": cost_frac,
                     "net": net, "turnover": turnover, "nav": nav})

    daily = pd.DataFrame(rows).set_index("datetime") if rows else \
        pd.DataFrame(columns=["gross", "cost", "net", "turnover", "nav"])
    return {"daily": daily, "final_nav": nav, "capital": float(capital)}
