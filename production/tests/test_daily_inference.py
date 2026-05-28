"""Unit tests for daily_inference helper functions (no mlflow / qlib needed)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from production.daily_inference import (
    _missing_dates,
    _group_by_handler_signature,
    _composite_and_calibrate,
    _handler_signature,
    _default_handler_cfg,
)


def test_missing_dates_returns_only_new():
    qlib_dates = pd.DatetimeIndex(pd.date_range("2026-05-20", periods=8, freq="B"))
    pred_dates = pd.DatetimeIndex(pd.date_range("2026-05-20", periods=5, freq="B"))
    out = _missing_dates(qlib_dates, pred_dates)
    # First 5 dates overlap, last 3 are new
    expected = list(pd.date_range("2026-05-27", periods=3, freq="B").date)
    assert out == expected


def test_missing_dates_empty_when_pred_caught_up():
    dates = pd.DatetimeIndex(pd.date_range("2026-05-20", periods=5, freq="B"))
    assert _missing_dates(dates, dates) == []


def test_missing_dates_empty_when_pred_ahead():
    qlib_dates = pd.DatetimeIndex(pd.date_range("2026-05-20", periods=3, freq="B"))
    pred_dates = pd.DatetimeIndex(pd.date_range("2026-05-20", periods=5, freq="B"))
    assert _missing_dates(qlib_dates, pred_dates) == []


def test_handler_signature_distinguishes_alpha158_vs_alpha360():
    cfg_158 = {"class": "Alpha158", "kwargs": {"start_time": "2020-01-01"}}
    cfg_360 = {"class": "Alpha360_OpenH", "kwargs": {"start_time": "2020-01-01"}}
    assert _handler_signature(cfg_158) != _handler_signature(cfg_360)


def test_handler_signature_ignores_segment_kwargs():
    """Two cfgs that differ only in start_time/end_time/instruments should
    have the same signature (since those get overridden at inference time)."""
    a = {"class": "Alpha158", "kwargs": {"start_time": "2020", "end_time": "2025", "instruments": "csi800"}}
    b = {"class": "Alpha158", "kwargs": {"start_time": "2021", "end_time": "2026", "instruments": "csi500"}}
    assert _handler_signature(a) == _handler_signature(b)


def test_group_by_handler_signature_collapses_alpha360_models():
    loaded = {
        "lgbm":  {"1d": ("model_l1", {"class": "Alpha158", "kwargs": {}})},
        "alstm": {"1d": ("model_a1", {"class": "Alpha360", "kwargs": {}}),
                  "5d": ("model_a5", {"class": "Alpha360", "kwargs": {}})},
        "tra":   {"1d": ("model_t1", {"class": "Alpha360", "kwargs": {}})},
    }
    groups = _group_by_handler_signature(loaded)
    assert len(groups) == 2
    sizes = sorted(len(g) for g in groups.values())
    assert sizes == [1, 3]


def test_composite_and_calibrate_returns_three_horizons():
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2026-05-25", periods=2, freq="B"),
         [f"S{i:03d}" for i in range(50)]],
        names=["datetime", "instrument"],
    )
    raw = pd.DataFrame({
        "lgbm_1d": rng.normal(size=len(idx)),
        "alstm_1d": rng.normal(size=len(idx)),
        "tra_1d": rng.normal(size=len(idx)),
        "lgbm_5d": rng.normal(size=len(idx)),
        "alstm_5d": rng.normal(size=len(idx)),
        "tra_5d": rng.normal(size=len(idx)),
        "lgbm_20d": rng.normal(size=len(idx)),
        "alstm_20d": rng.normal(size=len(idx)),
        "tra_20d": rng.normal(size=len(idx)),
    }, index=idx)
    cal = {"maps": {}, "meta": {}}
    out = _composite_and_calibrate(raw, cal)
    assert "score" in out.columns
    assert "consensus" in out.columns
    assert out["score"].notna().all()


def test_default_handler_cfg_returns_alpha360_for_nn_models():
    assert _default_handler_cfg("alstm")["class"] in ("Alpha360_OpenH", "Alpha360")
    assert _default_handler_cfg("tra")["class"] in ("Alpha360_OpenH", "Alpha360")


def test_default_handler_cfg_returns_alpha158_for_lgbm():
    assert _default_handler_cfg("lgbm")["class"] in ("Alpha158", "Alpha158_OpenH")
