# production/backtest/executability.py
"""A股 entry-open buyability. A name whose entry-day OPEN gaps to its 涨停 ceiling
can't be bought at open. Used to (a) gate top-k selection to buyable names and
(b) decompose whether the unbuyable (gapped-up) names are the winners we miss."""
from __future__ import annotations

# Force installed qlib ahead of the uncompiled ./qlib tree (only load_entry_ohlc
# touches qlib; the pure helpers below don't).
import sys as _sys
import sysconfig as _sysconfig
_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import numpy as np
import pandas as pd

from production.intraday.entry_rules import limit_up_price


def buyable_mask(ohlc: pd.DataFrame) -> pd.Series:
    """ohlc indexed (datetime, instrument) with columns [entry_open, prev_close]
    (entry_open = open of entry session d+1; prev_close = close of decision day d).
    True when entry open is strictly below the name's 涨停 ceiling (buy-at-open fills)."""
    idx = ohlc.index
    insts = idx.get_level_values("instrument")
    caps = pd.Series(
        [limit_up_price(i, pc) for i, pc in zip(insts, ohlc["prev_close"].to_numpy())],
        index=idx)
    return (ohlc["entry_open"] < caps - 1e-9).rename("buyable")


def gate_scores(scores: pd.Series, buyable: pd.Series) -> pd.Series:
    """Drop (datetime,instrument) entries that aren't buyable-at-open so downstream
    nlargest rolls to the next buyable rank. Unknown (missing) names are kept."""
    s = scores.dropna()
    if buyable is None or len(buyable) == 0:
        return s
    keep = buyable.reindex(s.index).fillna(True).astype(bool)
    return s[keep]


def selection_bias_split(scores: pd.Series, fwd_ret: pd.Series, buyable: pd.Series,
                         top_k: int, period: int) -> dict:
    """On each rebalance day (every `period` steps) take the UNGATED top-k by score,
    split into buyable-at-open vs unbuyable (gapped-up), collect realized open->open
    fwd returns. edge_missed = mean(unbuyable) - mean(buyable): >0 means the winners
    are exactly the ones a live trader can't buy."""
    s = scores.dropna()
    dates = sorted(s.index.get_level_values("datetime").unique())
    fwd_dates = set(fwd_ret.index.get_level_values("datetime").unique())
    buy_ret, miss_ret = [], []
    n_pick = n_miss = 0
    for i, d in enumerate(dates):
        if i % period != 0 or d not in fwd_dates:
            continue
        cross = s.xs(d, level="datetime").sort_values(ascending=False)
        r_d = fwd_ret.xs(d, level="datetime")
        try:
            b_d = buyable.xs(d, level="datetime")
        except KeyError:
            b_d = pd.Series(dtype=bool)
        for inst in list(cross.index[:top_k]):
            n_pick += 1
            ret = r_d.get(inst)
            if ret is None or pd.isna(ret):
                continue
            if bool(b_d.get(inst, True)):
                buy_ret.append(float(ret))
            else:
                miss_ret.append(float(ret)); n_miss += 1

    def _m(a):
        return float(np.mean(a)) if a else float("nan")
    return {
        "n_picks": n_pick, "n_unbuyable": n_miss,
        "unbuyable_pct": (n_miss / n_pick) if n_pick else 0.0,
        "buyable_mean_ret": _m(buy_ret), "unbuyable_mean_ret": _m(miss_ret),
        "buyable_n": len(buy_ret), "unbuyable_n": len(miss_ret),
        "edge_missed": _m(miss_ret) - _m(buy_ret),
    }


def load_entry_ohlc(instruments, start: str, end: str,
                    config_path: str = "production/configs/rolling_ensemble.yaml") -> pd.DataFrame:
    """Per decision day d, load the entry session's open/high/low + decision-day close
    (涨停 base). Aligned to d like load_fwd_returns (entry at d+1):
      entry_open=Ref($open,-1), entry_high=Ref($high,-1), entry_low=Ref($low,-1),
      prev_close=$close. Integration helper (qlib); not unit-tested."""
    from qlib.data.dataset.loader import QlibDataLoader
    from production.backtest.data import init_qlib_from_config
    init_qlib_from_config(config_path)
    fields = ["Ref($open,-1)", "Ref($high,-1)", "Ref($low,-1)", "$close"]
    names = ["entry_open", "entry_high", "entry_low", "prev_close"]
    df = QlibDataLoader(config={"feature": (fields, names)}).load(
        instruments=instruments, start_time=start, end_time=end)
    if df.index.names[0] == "instrument":
        df = df.swaplevel().sort_index()
    df.index = df.index.set_names(["datetime", "instrument"])
    return df.dropna(subset=["entry_open", "prev_close"])
