"""Offline intraday-execution simulator: replace next-open entry with rule-based
intraday entry for the factor-2model fixed/hold-5/5d top-k picks; compare net."""
from __future__ import annotations
import sys, sysconfig
_P = sysconfig.get_paths().get("purelib")
if _P and _P not in sys.path[:1]:
    sys.path.insert(0, _P)
from pathlib import Path
import numpy as np, pandas as pd

# Plausible same-day entry-price / open ratio. A multiplier outside this band can
# only come from a bad 5min print (zero/garbage bar) -> treat as glitch, fall back
# to the open baseline rather than book an impossible fill.
MULT_LO, MULT_HI = 0.5, 1.5


def enumerate_trades(scores: pd.Series, top_k: int = 5, period: int = 5) -> list[dict]:
    """Walk the fixed/period rebalance schedule; on each rebalance day pick top_k
    by score; map decision_date -> entry_date (next session) -> exit_date (+period
    sessions). Returns one dict per (rebalance, name)."""
    dates = sorted(scores.index.get_level_values("datetime").unique())
    out = []
    for step, i in enumerate(range(0, len(dates), period)):
        d = dates[i]
        if i + 1 >= len(dates):
            break
        entry = dates[i + 1]
        exit_i = min(i + 1 + period, len(dates) - 1)
        exit_ = dates[exit_i]
        cross = scores.xs(d, level="datetime").dropna().sort_values(ascending=False)
        for inst in list(cross.index[:top_k]):
            out.append({"rebalance_step": i, "decision_date": d, "entry_date": entry,
                        "exit_date": exit_, "instrument": inst})
    return out


def daily_open_adj(instruments, start, end,
                   config="production/configs/rolling_ensemble.yaml") -> pd.Series:
    """Adjusted daily $open per (datetime,instrument) via qlib (engine-consistent)."""
    from qlib.data.dataset.loader import QlibDataLoader
    from production.backtest.data import init_qlib_from_config
    init_qlib_from_config(config)
    px = QlibDataLoader(config={"feature": (["$open"], ["open"])}).load(
        instruments=instruments, start_time=start, end_time=end)
    s = px.iloc[:, 0] if isinstance(px, pd.DataFrame) else px
    if s.index.names[0] == "instrument":
        s = s.swaplevel().sort_index()
    s.index = s.index.set_names(["datetime", "instrument"])
    return s.sort_index()


def simulate(scores, *, rule, top_k=5, period=5, k=0.01, g=0.03,
             cost_bps=10.0) -> dict:
    """For each trade: entry_adj = open_adj(entry) * entry_multiplier(rule);
    ret = open_adj(exit)/entry_adj - 1 - cost; aggregate equal-weight per
    rebalance into a period-return series -> net metrics. rule='open' reproduces
    the open baseline (multiplier 1.0, no fetch)."""
    from production.intraday.entry_rules import entry_multiplier, is_buy_fillable
    from production.intraday.fetch_5min import fetch_5min, prev_close_raw
    trades = enumerate_trades(scores, top_k, period)
    insts = sorted({t["instrument"] for t in trades})
    dmin = min(t["entry_date"] for t in trades); dmax = max(t["exit_date"] for t in trades)
    opens = daily_open_adj(insts, str(dmin.date()), str(dmax.date()))
    per_rebalance: dict = {}
    step_date: dict = {}
    improve_bps: list = []
    n_filled = n_unfillable = n_gap_skip = n_no_open = n_fallback = n_glitch = 0
    for t in trades:
        oe = opens.get((t["entry_date"], t["instrument"]))
        ox = opens.get((t["exit_date"], t["instrument"]))
        if oe is None or ox is None or not (oe > 0):
            n_no_open += 1; continue
        mult = 1.0
        if rule != "open":
            ed = t["entry_date"].strftime("%Y-%m-%d")
            bars = fetch_5min(t["instrument"], ed, ed)
            pc = prev_close_raw(t["instrument"], ed)
            has_data = (bars is not None and not bars.empty
                        and float(bars["volume"].sum()) > 0
                        and float(bars["open"].max()) > 0)
            if not has_data or pc is None:
                n_fallback += 1                  # halt / no 5min / no prev_close -> degrade to open (mult 1.0)
            elif not is_buy_fillable(bars, pc, t["instrument"]):
                n_unfillable += 1; continue      # limit-up (一字封板): genuinely can't buy -> skip
            else:
                m = entry_multiplier(bars, pc, t["instrument"], rule=rule, k=k, g=g)
                if m is None:
                    n_gap_skip += 1; continue    # gap_cond don't-chase -> intentional skip
                if not (MULT_LO <= m <= MULT_HI):
                    n_glitch += 1                # bad print -> degrade to open baseline (mult 1.0)
                else:
                    mult = m
                    improve_bps.append((1.0 - mult) * 1e4)   # +bp = cheaper entry than open
        entry_adj = float(oe) * mult
        ret = float(ox) / entry_adj - 1 - cost_bps / 1e4
        per_rebalance.setdefault(t["rebalance_step"], []).append(ret)
        step_date[t["rebalance_step"]] = t["decision_date"]
        n_filled += 1
    periods = sorted(per_rebalance)
    idx = pd.DatetimeIndex([step_date[p] for p in periods])
    pr = pd.Series([float(np.mean(per_rebalance[p])) for p in periods], index=idx)
    eq = (1 + pr).cumprod()
    n = len(pr)
    ann = (eq.iloc[-1] ** (252 / (period * n)) - 1) if n and eq.iloc[-1] > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1).min()) if n else float("nan")
    by_year = {}
    for y in sorted({d.year for d in pr.index}):
        py = pr[pr.index.year == y]
        by_year[int(y)] = float((1 + py).prod() - 1) if len(py) else float("nan")
    n_trades = len(trades)
    return {"rule": rule, "net_cagr": ann,
            "calmar": (ann / abs(dd)) if dd else float("nan"),
            "max_dd": dd, "win": float((pr > 0).mean()) if n else float("nan"),
            "n_periods": n, "n_trades": n_trades, "n_filled": n_filled,
            "n_unfillable": n_unfillable, "n_gap_skip": n_gap_skip,
            "n_no_open": n_no_open, "n_fallback": n_fallback, "n_glitch": n_glitch,
            "unfillable_pct": (n_unfillable / n_trades) if n_trades else 0.0,
            "fallback_pct": (n_fallback / n_trades) if n_trades else 0.0,
            "glitch_pct": (n_glitch / n_trades) if n_trades else 0.0,
            "improve_bps_med": float(np.median(improve_bps)) if improve_bps else 0.0,
            "by_year": by_year}
