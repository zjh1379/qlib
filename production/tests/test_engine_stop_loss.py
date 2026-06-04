"""TDD for the optional per-name stop-loss in production/backtest/engine.run_backtest.

Contract:
  - stop_loss=None  -> identical behaviour to before (backward compatible).
  - stop_loss=x     -> a held name whose cumulative since-entry return breaches
                       -x is exited (weight -> 0 / cash) from the next day, so
                       it stops bleeding. A cratering name therefore hurts the
                       portfolio LESS with a stop than without one.
"""
from __future__ import annotations

import pandas as pd
import pytest

from production.backtest.engine import run_backtest
from production.backtest.rebalance import FixedPeriod


class _NoCost:
    def trade_cost(self, notional, is_buy=True):
        return 0.0


def _panel(ret_by_inst: dict[str, list[float]], start="2024-01-01") -> pd.Series:
    frames = []
    n = len(next(iter(ret_by_inst.values())))
    idx = pd.date_range(start, periods=n, freq="B")
    for inst, rets in ret_by_inst.items():
        mi = pd.MultiIndex.from_product([idx, [inst]], names=["datetime", "instrument"])
        frames.append(pd.Series(rets, index=mi))
    return pd.concat(frames).sort_index()


def _scores(insts, dates_from_fwd: pd.Series) -> pd.Series:
    # constant scores so FixedPeriod always picks the same top-k
    idx = dates_from_fwd.index
    val = {inst: float(len(insts) - i) for i, inst in enumerate(insts)}
    return pd.Series([val[i] for (_, i) in idx], index=idx, name="score")


def test_stop_loss_none_is_backward_compatible():
    fwd = _panel({"A": [0.01] * 8, "B": [-0.02] * 8})
    scores = _scores(["A", "B"], fwd)
    pol = FixedPeriod(top_k=2, period=100)  # rebalance once -> hold throughout
    base = run_backtest(scores, fwd, pol, _NoCost(), capital=100_000.0)
    none = run_backtest(scores, fwd, pol, _NoCost(), capital=100_000.0, stop_loss=None)
    assert base["final_nav"] == pytest.approx(none["final_nav"])
    assert base["daily"]["net"].tolist() == pytest.approx(none["daily"]["net"].tolist())


def test_no_crater_means_stop_changes_nothing():
    fwd = _panel({"A": [0.01] * 8, "B": [0.002] * 8})  # nobody breaches -10%
    scores = _scores(["A", "B"], fwd)
    pol = FixedPeriod(top_k=2, period=100)
    no = run_backtest(scores, fwd, pol, _NoCost(), capital=100_000.0)
    st = run_backtest(scores, fwd, pol, _NoCost(), capital=100_000.0, stop_loss=0.10)
    assert no["final_nav"] == pytest.approx(st["final_nav"])


def test_crater_is_capped_by_stop():
    # B craters -4%/day: cum breaches -10% after day 3 -> stop exits it.
    fwd = _panel({"A": [0.005] * 10, "B": [-0.04] * 10})
    scores = _scores(["A", "B"], fwd)
    pol = FixedPeriod(top_k=2, period=100)
    no = run_backtest(scores, fwd, pol, _NoCost(), capital=100_000.0)
    st = run_backtest(scores, fwd, pol, _NoCost(), capital=100_000.0, stop_loss=0.10)
    # stop-loss must leave the portfolio strictly better off than riding B down
    assert st["final_nav"] > no["final_nav"]
    # and B must stop contributing: late-period daily net should be ~ A-only
    late_st = st["daily"]["net"].iloc[-1]
    assert late_st == pytest.approx(0.5 * 0.005, abs=1e-9)  # only A (0.5 wt), B in cash
