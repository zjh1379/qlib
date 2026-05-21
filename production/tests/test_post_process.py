import pytest

import pandas as pd

from production.post_process import ewma_smooth, cost_adjust


def test_ewma_smooth_first_day_passthrough():
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-05-15", "2026-05-15"]),
            "instrument": ["SH600000", "SH600001"],
            "score": [0.10, 0.20],
        }
    ).set_index(["datetime", "instrument"])

    out = ewma_smooth(df, alpha=0.5)
    # First observation per stock should equal raw
    assert out.loc[(pd.Timestamp("2026-05-15"), "SH600000"), "score"] == pytest.approx(0.10)
    assert out.loc[(pd.Timestamp("2026-05-15"), "SH600001"), "score"] == pytest.approx(0.20)


def test_ewma_smooth_second_day_blends_previous():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-05-15"), "SH600000"),
            (pd.Timestamp("2026-05-16"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame({"score": [0.10, 0.30]}, index=idx)
    out = ewma_smooth(df, alpha=0.5)
    # day-1 score = 0.5*0.30 + 0.5*0.10 = 0.20
    assert out.loc[(pd.Timestamp("2026-05-16"), "SH600000"), "score"] == pytest.approx(0.20)


def test_cost_adjust_subtracts_turnover_cost():
    returns = pd.Series([0.02, 0.01, -0.005])
    turnover = pd.Series([0.20, 0.05, 0.15])
    bps = 10  # 0.1%
    adjusted = cost_adjust(returns, turnover, bps=bps)
    expected = returns - turnover * (bps / 10_000)
    pd.testing.assert_series_equal(adjusted, expected, check_names=False)
