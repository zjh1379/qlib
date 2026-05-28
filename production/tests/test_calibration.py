"""Tests for production/calibration.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from production.calibration import (
    fit_calibration,
    apply_calibration,
    load_calibration,
    save_calibration,
    _composite_score,
)


def _toy_pred_df(seed: int = 0, n_dates: int = 30, n_inst: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="B")
    inst = [f"S{i:03d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dates, inst], names=["datetime", "instrument"])
    return pd.DataFrame({
        "lgbm_5d": rng.normal(size=len(idx)),
        "alstm_5d": rng.normal(size=len(idx)),
        "tra_5d": rng.normal(size=len(idx)),
        "lgbm_1d": rng.normal(size=len(idx)),
        "alstm_1d": rng.normal(size=len(idx)),
        "tra_1d": rng.normal(size=len(idx)),
    }, index=idx)


def _toy_label_df(pred_df: pd.DataFrame, signal: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    out = {}
    for h in ("1d", "5d"):
        comp = _composite_score(pred_df, h)
        out[f"label_{h}"] = comp * signal + rng.normal(scale=0.5, size=len(comp))
    return pd.DataFrame(out, index=pred_df.index)


def test_fit_calibration_returns_dict_keyed_by_horizon():
    pred = _toy_pred_df()
    label = _toy_label_df(pred)
    cal = fit_calibration(pred, label, horizons=("1d", "5d"))
    assert set(cal.keys()) == {"1d", "5d"}


def test_isotonic_is_monotonic_increasing():
    pred = _toy_pred_df()
    label = _toy_label_df(pred, signal=1.0)
    cal = fit_calibration(pred, label, horizons=("5d",))
    iso = cal["5d"]
    xs = np.linspace(-100, 100, 50)
    ys = iso.predict(xs)
    assert (np.diff(ys) >= -1e-9).all(), "isotonic must be non-decreasing"


def test_apply_calibration_returns_series_with_same_index():
    pred = _toy_pred_df()
    label = _toy_label_df(pred)
    cal = fit_calibration(pred, label, horizons=("5d",))
    comp = _composite_score(pred, "5d")
    out = apply_calibration(comp, cal["5d"])
    assert isinstance(out, pd.Series)
    assert out.index.equals(comp.index)
    assert out.notna().all()


def test_clip_out_of_bounds():
    pred = _toy_pred_df()
    label = _toy_label_df(pred)
    cal = fit_calibration(pred, label, horizons=("5d",))
    out = apply_calibration(pd.Series([9999.0, -9999.0]), cal["5d"])
    assert out.notna().all()
    assert np.isfinite(out).all()


def test_small_sample_skipped():
    # 20 dates × 50 instruments = 1000 rows >= MIN_SAMPLES (100)
    # Need to force small sample by using few dates
    pred = _toy_pred_df(n_dates=2, n_inst=10)  # 20 rows
    label = _toy_label_df(pred)
    cal = fit_calibration(pred, label, horizons=("5d",))
    assert "5d" not in cal


def test_save_load_round_trip(tmp_path):
    pred = _toy_pred_df()
    label = _toy_label_df(pred)
    cal = fit_calibration(pred, label, horizons=("1d", "5d"))
    p = tmp_path / "cal.pkl"
    save_calibration(cal, p, meta={"trained_at": "2026-05-22"})
    loaded = load_calibration(p)
    assert set(loaded["maps"].keys()) == {"1d", "5d"}
    assert loaded["meta"]["trained_at"] == "2026-05-22"


def test_apply_calibration_handles_all_nan_input():
    pred = _toy_pred_df()
    label = _toy_label_df(pred)
    cal = fit_calibration(pred, label, horizons=("5d",))
    out = apply_calibration(pd.Series([np.nan, np.nan, np.nan]), cal["5d"])
    assert out.isna().all()


def test_load_calibration_returns_empty_when_missing(tmp_path):
    p = tmp_path / "nonexistent.pkl"
    loaded = load_calibration(p)
    assert loaded == {"maps": {}, "meta": {}}


def test_fit_calibration_skips_missing_horizon():
    pred = _toy_pred_df()
    label = _toy_label_df(pred)  # only 1d and 5d
    cal = fit_calibration(pred, label, horizons=("1d", "5d", "20d"))
    # 20d should be skipped (no label)
    assert "20d" not in cal
    assert "1d" in cal and "5d" in cal
