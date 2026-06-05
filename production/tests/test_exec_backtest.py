import pandas as pd, pytest
from production.intraday.exec_backtest import enumerate_trades


def _scores():
    days = pd.date_range("2024-01-01", periods=12, freq="B")
    rows = []
    for d in days:
        for i, inst in enumerate(["A", "B", "C"]):
            rows.append(((d, inst), 3 - i))  # A>B>C every day
    s = pd.Series(dict(rows)); s.index = s.index.set_names(["datetime", "instrument"]); return s


def test_enumerate_trades_fixed_5d_topk2():
    s = _scores()
    trades = enumerate_trades(s, top_k=2, period=5)
    # rebalance at step 0 and 5; entry = NEXT trading day's open; top-2 = A,B
    assert {t["instrument"] for t in trades} == {"A", "B"}
    t0 = [t for t in trades if t["rebalance_step"] == 0]
    assert len(t0) == 2
    # entry_date is the trading day AFTER the decision day; exit_date 5 sessions later
    assert t0[0]["entry_date"] > t0[0]["decision_date"]
    assert t0[0]["exit_date"] > t0[0]["entry_date"]
