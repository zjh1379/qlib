import pandas as pd
from production.backtest.rebalance import Daily
from production.backtest.rebalance import FixedPeriod
from production.backtest.rebalance import Banded


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


def test_banded_keeps_held_name_inside_exit_band():
    p = Banded(top_k=2, exit_k=4)
    # currently hold A,B (equal weight). New scores rank A=1,B=3,C=2,D=4,E=5.
    current = pd.Series({"A": 0.5, "B": 0.5})
    scores = pd.Series({"A": 5.0, "C": 4.0, "B": 3.0, "D": 2.0, "E": 1.0})
    w = p.target_weights(scores, current)
    # A (rank1) and B (rank3 <= exit_k 4) are both kept -> no churn into C.
    assert set(w.index) == {"A", "B"}


def test_banded_drops_held_name_outside_exit_band():
    p = Banded(top_k=2, exit_k=3)
    current = pd.Series({"A": 0.5, "B": 0.5})
    # B falls to rank 4 (> exit_k 3) -> dropped, replaced by best new (C rank2).
    scores = pd.Series({"A": 5.0, "C": 4.0, "D": 3.0, "B": 2.0, "E": 1.0})
    w = p.target_weights(scores, current)
    assert "B" not in w.index
    assert set(w.index) == {"A", "C"}


def test_banded_turnover_lower_than_daily():
    import numpy as np
    from production.backtest.engine import run_backtest
    from production.backtest.rebalance import Daily
    from production.backtest.costs import CostModel
    dates = pd.bdate_range("2024-01-02", periods=30)
    stocks = [f"S{i}" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    g = np.random.default_rng(1)
    # persistent-ish scores so banding can hold
    base = {s: g.normal(0, 1) for s in stocks}
    vals = [base[s] + g.normal(0, 0.2) for (_, s) in idx]
    scores = pd.Series(vals, index=idx)
    fwd = pd.Series(g.normal(0, 0.01, len(idx)), index=idx)
    daily = run_backtest(scores, fwd, Daily(top_k=5), CostModel(), 1e5)["daily"]
    banded = run_backtest(scores, fwd, Banded(top_k=5, exit_k=10), CostModel(), 1e5)["daily"]
    assert banded["turnover"].mean() < daily["turnover"].mean()
