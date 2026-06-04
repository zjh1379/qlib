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
    exposure: pd.Series | None = None,
    stop_loss: float | None = None,
) -> dict:
    """
    scores: Series (datetime,instrument)->score, OR DataFrame with `score_col`.
    fwd_ret: Series (datetime,instrument)->open(d+2)/open(d+1)-1.
    stop_loss: optional per-name stop. When set (e.g. 0.10), a held name whose
        cumulative since-entry return breaches -stop_loss is exited to cash from
        the next day (re-entry only on the next rebalance). None = no stop
        (identical to the original behaviour).
    Returns dict: {"daily": DataFrame[gross,cost,net,turnover,nav], "final_nav", "capital"}.
    """
    s = scores[score_col] if isinstance(scores, pd.DataFrame) else scores
    fwd_dates = set(fwd_ret.index.get_level_values("datetime").unique())
    dates = [d for d in sorted(s.index.get_level_values("datetime").unique()) if d in fwd_dates]

    current = pd.Series(dtype=float)
    entry_cum: dict = {}  # held name -> cumulative since-entry return factor (stop-loss)
    nav = float(capital)
    rows = []
    for i, d in enumerate(dates):
        s_d = s.xs(d, level="datetime")
        cost_frac = 0.0
        turnover = 0.0
        if policy.should_rebalance(i):
            target = policy.target_weights(s_d, current)
            if exposure is not None and len(target):
                e = exposure.asof(d) if len(exposure) and d >= exposure.index[0] else 1.0
                if pd.isna(e):
                    e = 1.0
                e = float(min(1.0, max(0.0, e)))
                target = target * e
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
            if stop_loss is not None:
                # rebalance = fresh equal-weight entries; reset since-entry tracking
                entry_cum = {inst: 1.0 for inst in current.index}
        r_d = fwd_ret.xs(d, level="datetime")
        gross = float((current * r_d.reindex(current.index).fillna(0.0)).sum())
        net = gross - cost_frac
        nav *= (1 + net)
        rows.append({"datetime": d, "gross": gross, "cost": cost_frac,
                     "net": net, "turnover": turnover, "nav": nav})
        if stop_loss is not None and len(current):
            # update since-entry cum returns; exit (-> cash) names that breached -stop_loss
            rr = r_d.reindex(current.index).fillna(0.0)
            stopped = []
            for inst in current.index:
                if current[inst] == 0:
                    continue
                entry_cum[inst] = entry_cum.get(inst, 1.0) * (1.0 + float(rr[inst]))
                if entry_cum[inst] - 1.0 <= -stop_loss:
                    stopped.append(inst)
            if stopped:
                current = current.copy()
                current.loc[stopped] = 0.0
                for inst in stopped:
                    entry_cum.pop(inst, None)

    daily = pd.DataFrame(rows).set_index("datetime") if rows else \
        pd.DataFrame(columns=["gross", "cost", "net", "turnover", "nav"])
    return {"daily": daily, "final_nav": nav, "capital": float(capital)}
