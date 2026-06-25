# production/backtest/etf_timing.py
"""Tiny single-asset ETF market-timing sim: hold one broad ETF, scale exposure in
[0,1] by the trend overlay, charge ETF cost only when exposure flips. NOT the
cross-sectional stock engine — a one-asset daily ledger compatible with net_metrics."""
from __future__ import annotations

import pandas as pd

from production.backtest.costs import CostModel


def simulate_etf_timing(etf_ret: pd.Series, exposure: pd.Series,
                        cost: CostModel, capital: float = 10_000.0) -> pd.DataFrame:
    """etf_ret: daily ETF return by datetime. exposure: target weight [0,1] by datetime
    (decided from trailing data). Position(d) = exposure(d-1) (acted next day, no
    lookahead); pnl(d)=pos(d)*etf_ret(d); cost on |pos change| notional when flipping.
    Returns daily ledger [gross,cost,net,turnover,nav]."""
    etf_ret = etf_ret.sort_index()
    pos = exposure.reindex(etf_ret.index).ffill().shift(1).fillna(0.0).clip(0.0, 1.0)
    rows = []
    nav = float(capital)
    prev = 0.0
    for d in etf_ret.index:
        p = float(pos.loc[d])
        gross = p * float(etf_ret.loc[d])
        dpos = abs(p - prev)
        cost_yuan = cost.trade_cost(dpos * nav, is_buy=(p >= prev)) if dpos > 1e-9 else 0.0
        cost_frac = cost_yuan / nav if nav > 0 else 0.0
        net = gross - cost_frac
        nav *= (1 + net)
        rows.append({"datetime": d, "gross": gross, "cost": cost_frac,
                     "net": net, "turnover": dpos, "nav": nav})
        prev = p
    return pd.DataFrame(rows).set_index("datetime")
