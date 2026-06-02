# production/tests/test_engine.py
import pandas as pd
import pytest
from production.backtest.engine import run_backtest
from production.backtest.rebalance import Daily
from production.backtest.costs import CostModel


def _series(d):
    idx = pd.MultiIndex.from_tuples(list(d.keys()), names=["datetime", "instrument"])
    return pd.Series(list(d.values()), index=idx)


def test_cost_reconciliation_first_day():
    # 2 dates, 2 stocks; Daily top_k=1 picks A both days.
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    scores = _series({(dates[0], "A"): 1.0, (dates[0], "B"): 0.0,
                      (dates[1], "A"): 1.0, (dates[1], "B"): 0.0})
    fwd = _series({(dates[0], "A"): 0.01, (dates[0], "B"): 0.0,
                   (dates[1], "A"): 0.02, (dates[1], "B"): 0.0})
    cm = CostModel(commission_bps=2.5, commission_min_yuan=5.0,
                   stamp_bps=5.0, transfer_bps=0.1, slippage_bps=5.0)
    res = run_backtest(scores, fwd, Daily(top_k=1), cm, capital=100_000.0)
    day0 = res["daily"].iloc[0]
    # Day0: buy A notional=100000 -> cost = max(25,5)+0+1+50 = 76 -> 0.00076
    assert day0["cost"] == pytest.approx(76 / 100_000, rel=1e-9)
    assert day0["turnover"] == pytest.approx(0.5, rel=1e-9)  # 0.5*|+1.0| (B has 0 delta)
    assert day0["gross"] == pytest.approx(0.01, rel=1e-9)
    assert day0["net"] == pytest.approx(0.01 - 0.00076, rel=1e-9)


def test_no_trade_second_day_when_holding_same():
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    scores = _series({(dates[0], "A"): 1.0, (dates[1], "A"): 1.0})
    fwd = _series({(dates[0], "A"): 0.01, (dates[1], "A"): 0.02})
    res = run_backtest(scores, fwd, Daily(top_k=1), CostModel(), capital=100_000.0)
    # Day1 holds same A -> delta 0 -> cost 0, turnover 0
    assert res["daily"].iloc[1]["cost"] == 0.0
    assert res["daily"].iloc[1]["turnover"] == 0.0
    assert res["daily"].iloc[1]["gross"] == pytest.approx(0.02, rel=1e-9)


def test_iterates_only_dates_with_fwd_ret():
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
    scores = _series({(d, "A"): 1.0 for d in dates})
    # fwd_ret missing the last date (no future price)
    fwd = _series({(dates[0], "A"): 0.01, (dates[1], "A"): 0.02})
    res = run_backtest(scores, fwd, Daily(top_k=1), CostModel(), capital=100_000.0)
    assert len(res["daily"]) == 2  # last date dropped


def test_constant_exposure_scales_gross_linearly():
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    scores = _series({(dates[0], "A"): 1.0, (dates[1], "A"): 1.0})
    fwd = _series({(dates[0], "A"): 0.01, (dates[1], "A"): 0.02})
    full = run_backtest(scores, fwd, Daily(top_k=1), CostModel(), 1e5)["daily"]
    exp = pd.Series(0.5, index=pd.Index(dates, name="datetime"))
    half = run_backtest(scores, fwd, Daily(top_k=1), CostModel(), 1e5, exposure=exp)["daily"]
    # second day holds same name -> no cost; gross should be exactly half
    assert half.iloc[1]["gross"] == pytest.approx(0.5 * full.iloc[1]["gross"], rel=1e-9)


def test_zero_exposure_no_gross():
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    scores = _series({(dates[0], "A"): 1.0, (dates[1], "A"): 1.0})
    fwd = _series({(dates[0], "A"): 0.05, (dates[1], "A"): 0.05})
    exp = pd.Series(0.0, index=pd.Index(dates, name="datetime"))
    res = run_backtest(scores, fwd, Daily(top_k=1), CostModel(), 1e5, exposure=exp)["daily"]
    assert res["gross"].abs().sum() == pytest.approx(0.0)


def test_exposure_reduces_drawdown_on_bear():
    import numpy as np
    dates = pd.bdate_range("2024-01-02", periods=60)
    stocks = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    scores = pd.Series(1.0, index=idx)
    # bear: -1%/day for second half
    rets = {d: (0.005 if i < 30 else -0.02) for i, d in enumerate(dates)}
    fwd = pd.Series([rets[d] for (d, _) in idx], index=idx)
    base = run_backtest(scores, fwd, Daily(top_k=2), CostModel(), 1e5)["daily"]
    # exposure: full first half, 0 in bear
    exp = pd.Series([1.0 if i < 30 else 0.0 for i in range(len(dates))],
                    index=pd.Index(dates, name="datetime"))
    over = run_backtest(scores, fwd, Daily(top_k=2), CostModel(), 1e5, exposure=exp)["daily"]
    base_dd = ((1 + base["net"]).cumprod() / (1 + base["net"]).cumprod().cummax() - 1).min()
    over_dd = ((1 + over["net"]).cumprod() / (1 + over["net"]).cumprod().cummax() - 1).min()
    assert over_dd > base_dd  # overlay drawdown is less negative
