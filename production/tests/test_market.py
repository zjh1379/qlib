# production/tests/test_market.py
import pandas as pd
import pytest
from production.backtest.market import mean_market_return, returns_to_close, MKT_RET_EXPR


def _ser(d):
    idx = pd.MultiIndex.from_tuples(list(d.keys()), names=["datetime", "instrument"])
    return pd.Series(list(d.values()), index=idx)


def test_mkt_ret_expr_is_trailing():
    # MUST be trailing (Ref +1 = yesterday), never forward (negative Ref)
    assert MKT_RET_EXPR == "$close / Ref($close, 1) - 1"


def test_mean_market_return_equal_weight():
    t1, t2 = pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")
    s = _ser({(t1, "A"): 0.02, (t1, "B"): 0.04, (t2, "A"): -0.01, (t2, "B"): -0.03})
    m = mean_market_return(s)
    assert m.loc[t1] == pytest.approx(0.03)
    assert m.loc[t2] == pytest.approx(-0.02)


def test_returns_to_close_cumprod():
    t1, t2 = pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")
    c = returns_to_close(pd.Series([0.1, -0.1], index=[t1, t2]))
    assert c.loc[t1] == pytest.approx(1.1)
    assert c.loc[t2] == pytest.approx(1.1 * 0.9)
