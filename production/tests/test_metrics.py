import numpy as np
import pandas as pd
import pytest

from production.metrics import compute_scorecard, regime_split, paired_ttest


def _mk_pred_label():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-02", "2024-12-31", freq="B")
    stocks = [f"SH60{i:04d}" for i in range(50)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    true = rng.normal(0, 0.02, size=len(idx))
    noise = rng.normal(0, 0.01, size=len(idx))
    pred = pd.Series(true * 0.5 + noise, index=idx, name="score")
    label = pd.Series(true, index=idx, name="label")
    return pred, label


def test_scorecard_returns_eight_keys():
    pred, label = _mk_pred_label()
    out = compute_scorecard(pred, label, top_k=10, bps=10)
    expected_keys = {
        "ic_mean", "ric_mean", "icir",
        "top_bottom_spread_monthly",
        "annual_excess_return", "ir", "max_drawdown",
        "daily_turnover",
    }
    assert expected_keys.issubset(out.keys())


def test_scorecard_ic_in_reasonable_range():
    pred, label = _mk_pred_label()
    out = compute_scorecard(pred, label, top_k=10, bps=10)
    assert out["ic_mean"] > 0


def test_regime_split_returns_segments():
    pred, label = _mk_pred_label()
    segments = regime_split(pred, label, segments=[("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31")])
    assert len(segments) == 2
    for seg_name, seg_metrics in segments.items():
        assert "ir" in seg_metrics


def test_paired_ttest_runs():
    a = pd.Series(np.random.normal(0.001, 0.02, 100))
    b = pd.Series(np.random.normal(0.000, 0.02, 100))
    t, p = paired_ttest(a, b)
    assert isinstance(t, float)
    assert 0.0 <= p <= 1.0
