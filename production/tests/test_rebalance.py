import pandas as pd
from production.backtest.rebalance import Daily


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
