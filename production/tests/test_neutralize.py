import numpy as np
import pandas as pd
import pytest
from production.neutralize import neutralize


def _scores(dates, stocks, vals):
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    return pd.Series(vals, index=idx)


def test_sector_neutral_zeros_each_sector_mean_per_day():
    dates = [pd.Timestamp("2024-01-02")]
    stocks = ["A", "B", "C", "D"]
    s = _scores(dates, stocks, [1.0, 3.0, 10.0, 20.0])
    sector = pd.Series({"A": "x", "B": "x", "C": "y", "D": "y"})
    out = neutralize(s, sector=sector)
    day = out.xs(dates[0], level="datetime")
    # within each sector, mean removed -> sector means are ~0
    assert day[["A", "B"]].mean() == pytest.approx(0.0, abs=1e-9)
    assert day[["C", "D"]].mean() == pytest.approx(0.0, abs=1e-9)


def test_returns_same_index():
    dates = pd.bdate_range("2024-01-02", periods=3)
    stocks = ["A", "B", "C"]
    s = _scores(dates, stocks, np.arange(9, dtype=float))
    out = neutralize(s, size=pd.Series({"A": 1.0, "B": 2.0, "C": 3.0}))
    assert out.index.equals(s.index)
    assert out.name == s.name


def test_missing_sector_treated_as_unknown_not_dropped():
    dates = [pd.Timestamp("2024-01-02")]
    stocks = ["A", "B", "C"]
    s = _scores(dates, stocks, [1.0, 3.0, 10.0])
    sector = pd.Series({"A": "x", "B": "x"})  # C missing from the map
    out = neutralize(s, sector=sector)
    day = out.xs(dates[0], level="datetime")
    assert day.notna().all()                       # C not dropped to NaN
    assert day["C"] == pytest.approx(0.0, abs=1e-9)  # C alone in UNKNOWN group -> demeaned to 0
