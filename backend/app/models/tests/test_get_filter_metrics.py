"""Tests for app.core.qlib_adapter.get_filter_metrics.

These tests use a real qlib data store (the conda-env cn_data_bs setup) and
sanity-check the shapes + a few computed values. They run only if qlib data is
available; otherwise skip gracefully.
"""
from datetime import date

import pytest

from app.core.qlib_adapter import (
    get_filter_metrics,
    get_latest_close_prices,
    init_qlib_once,
)


@pytest.fixture(scope="module")
def qlib_ready():
    try:
        init_qlib_once()
    except Exception as exc:
        pytest.skip(f"qlib not initializable in this environment: {exc}")


def test_empty_symbols_returns_empty(qlib_ready):
    assert get_filter_metrics([], end_date=None) == {}


def test_known_symbol_shape(qlib_ready):
    # Use a stable, liquid CSI300 symbol that should have data through the
    # qlib data store's last calendar day.
    out = get_filter_metrics(["SH600519"], end_date=None)
    if "SH600519" not in out:
        pytest.skip("SH600519 not present in this qlib data store")
    row = out["SH600519"]
    assert set(row.keys()) >= {
        "last_close",
        "pct_change_1d",
        "pct_change_3d",
        "pct_change_5d",
        "pct_change_10d",
        "pct_change_20d",
        "amplitude",
        "vol_ratio",
        "is_new_high_20d",
        "is_new_high_60d",
        "is_new_high_120d",
    }
    assert row["last_close"] > 0
    # Amplitude is (high - low) / prev_close, typically 0..0.20 for blue chips
    assert 0 <= row["amplitude"] < 1.0


def test_consistency_with_latest_close(qlib_ready):
    syms = ["SH600519", "SH600887"]
    metrics = get_filter_metrics(syms, end_date=None)
    closes = get_latest_close_prices(syms)
    for s in syms:
        if s in metrics and s in closes:
            assert metrics[s]["last_close"] == pytest.approx(closes[s], rel=1e-6)
