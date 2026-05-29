import pandas as pd
from production.backtest.rebalance import Daily
from production.backtest.rebalance import FixedPeriod


def test_daily_always_rebalances():
    p = Daily(top_k=2)
    assert p.should_rebalance(0) is True
    assert p.should_rebalance(7) is True


def test_daily_equal_weight_top_k():
    scores = pd.Series({"A": 0.9, "B": 0.5, "C": 0.1})
    w = Daily(top_k=2).target_weights(scores, pd.Series(dtype=float))
    assert set(w.index) == {"A", "B"}
    assert w["A"] == 0.5 and w["B"] == 0.5
    assert w.sum() == 1.0


def test_fixed_period_rebalances_every_n_steps():
    p = FixedPeriod(top_k=2, period=5)
    assert p.should_rebalance(0) is True
    assert p.should_rebalance(1) is False
    assert p.should_rebalance(5) is True
    assert p.should_rebalance(10) is True


def test_fixed_period_turnover_lower_than_daily_in_engine():
    import pandas as pd
    from production.backtest.engine import run_backtest
    from production.backtest.rebalance import Daily
    from production.backtest.costs import CostModel
    import numpy as np
    dates = pd.bdate_range("2024-01-02", periods=20)
    stocks = [f"S{i}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    g = np.random.default_rng(0)
    scores = pd.Series(g.normal(0, 1, len(idx)), index=idx)
    fwd = pd.Series(g.normal(0, 0.01, len(idx)), index=idx)
    daily = run_backtest(scores, fwd, Daily(top_k=3), CostModel(), 1e5)["daily"]
    fixed = run_backtest(scores, fwd, FixedPeriod(top_k=3, period=5), CostModel(), 1e5)["daily"]
    assert fixed["turnover"].mean() < daily["turnover"].mean()
