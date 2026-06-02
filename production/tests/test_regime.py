# production/tests/test_regime.py
import numpy as np
import pandas as pd
import pytest
from production.backtest.regime import compute_exposure


def _rising(n=200):
    return pd.Series(np.arange(1, n + 1, dtype=float), index=pd.bdate_range("2020-01-01", periods=n))


def _falling(n=200):
    return pd.Series(np.arange(n, 0, -1, dtype=float), index=pd.bdate_range("2020-01-01", periods=n))


def test_uptrend_full_exposure():
    # n=100 keeps the series steeply above its 20d MA (ratio ~0.105 >> 0.05),
    # so the graded exposure saturates at 1.0. (A gentle n=200 linear ramp sits
    # at ratio ~0.0499, just below saturation — a precision-boundary artifact.)
    e = compute_exposure(_rising(100), ma_window=20, band=0.10)
    assert e.iloc[-1] == pytest.approx(1.0)


def test_downtrend_floor_zero():
    e = compute_exposure(_falling(), ma_window=20, band=0.10, min_exposure=0.0)
    assert e.iloc[-1] == pytest.approx(0.0)


def test_warmup_is_full_exposure():
    e = compute_exposure(_rising(10), ma_window=20)
    assert (e == 1.0).all()


def test_no_lookahead_changing_future_doesnt_change_past():
    base = pd.Series([10.0] * 60 + [11, 12, 13, 14, 15],
                     index=pd.bdate_range("2020-01-01", periods=65), dtype=float)
    e1 = compute_exposure(base, ma_window=20, band=0.10)
    mod = base.copy(); mod.iloc[-1] = 999.0
    e2 = compute_exposure(mod, ma_window=20, band=0.10)
    assert e1.iloc[:-1].equals(e2.iloc[:-1])


def test_vol_target_compresses_exposure():
    # flat-but-jumpy series above its MA: trend ~full, but high vol -> vol_target cuts it
    rng = np.random.default_rng(0)
    px = pd.Series(100 + np.cumsum(rng.normal(0, 3, 200)),
                   index=pd.bdate_range("2020-01-01", periods=200))
    e_trend = compute_exposure(px, ma_window=20, band=0.10)
    e_vol = compute_exposure(px, ma_window=20, band=0.10, vol_target=0.05, vol_window=20)
    assert e_vol.iloc[-1] <= e_trend.iloc[-1] + 1e-9
