"""TDD for production/trend_filter.py — the trend-aware ranking gate.

These tests are OFFLINE: they do NOT import qlib. We inject small synthetic
price frames / precomputed trend-feature frames so the suite is fast and
deterministic. They assert the four contract properties required of the gate:

  (1) output index is a subset of the input index (gate only *removes* rows);
  (2) names in a constructed DOWNTREND are removed by the gate;
  (3) names in a constructed UPTREND are kept by the gate;
  (4) no-lookahead: the decision at date D depends only on prices <= D
      (truncating the frame after D does not change D's verdict).

Run from repo root:
  F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_trend_filter.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from production.trend_filter import (
    apply_trend_filter,
    compute_trend_features,
    TREND_FEATURE_EXPRS,
    TREND_FEATURE_NAMES,
)


# --------------------------------------------------------------------------- #
# Synthetic price helpers                                                      #
# --------------------------------------------------------------------------- #
def _price_panel(series_by_inst: dict[str, np.ndarray], start: str = "2024-01-01") -> pd.Series:
    """Build a (datetime, instrument) close Series from {inst: 1d-array}."""
    frames = []
    for inst, vals in series_by_inst.items():
        idx = pd.date_range(start, periods=len(vals), freq="B")
        frames.append(
            pd.Series(
                np.asarray(vals, dtype=float),
                index=pd.MultiIndex.from_product([idx, [inst]], names=["datetime", "instrument"]),
                name="close",
            )
        )
    return pd.concat(frames).sort_index()


def _uptrend(n: int = 90, start: float = 10.0, drift: float = 0.01) -> np.ndarray:
    # strictly compounding up -> MA5>MA10>MA20, close>MA20, momentum>0
    return start * (1.0 + drift) ** np.arange(n)


def _downtrend(n: int = 90, start: float = 50.0, drift: float = -0.01) -> np.ndarray:
    # strictly compounding down -> close<MA20 (falling knife), momentum<0
    return start * (1.0 + drift) ** np.arange(n)


def _flat(n: int = 90, level: float = 20.0) -> np.ndarray:
    return np.full(n, level, dtype=float)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def panel() -> pd.Series:
    return _price_panel(
        {
            "UP": _uptrend(),       # clean uptrend
            "DOWN": _downtrend(),   # clean downtrend / falling knife
            "FLAT": _flat(),        # sideways
        }
    )


@pytest.fixture
def features(panel) -> pd.DataFrame:
    """Precomputed trailing trend-feature frame (offline; no qlib)."""
    return compute_trend_features(panel)


@pytest.fixture
def scores(panel) -> pd.Series:
    """Equal raw scores for every (date, inst) so the gate alone decides."""
    s = pd.Series(1.0, index=panel.index, name="score")
    return s


# --------------------------------------------------------------------------- #
# compute_trend_features sanity                                                #
# --------------------------------------------------------------------------- #
def test_feature_columns_match_declared_names(features):
    for name in TREND_FEATURE_NAMES:
        assert name in features.columns
    assert len(TREND_FEATURE_EXPRS) == len(TREND_FEATURE_NAMES)


def test_uptrend_features_true_after_warmup(features):
    up = features.xs("UP", level="instrument").dropna()
    last = up.iloc[-1]
    assert bool(last["close_gt_ma20"]) is True
    assert bool(last["close_gt_ma60"]) is True
    assert bool(last["ma_aligned"]) is True
    assert last["momentum"] > 0


def test_downtrend_features_false_after_warmup(features):
    dn = features.xs("DOWN", level="instrument").dropna()
    last = dn.iloc[-1]
    assert bool(last["close_gt_ma20"]) is False
    assert bool(last["ma_aligned"]) is False
    assert last["momentum"] < 0


# --------------------------------------------------------------------------- #
# (1) output index subset of input index                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["none", "soft", "strict"])
def test_output_index_subset_of_input(scores, features, mode):
    out = apply_trend_filter(scores, features, mode=mode)
    assert isinstance(out, pd.Series)
    assert out.index.isin(scores.index).all()
    # gate must never invent (date, inst) pairs not present in the input
    assert set(out.index) <= set(scores.index)


def test_none_mode_is_passthrough(scores, features):
    out = apply_trend_filter(scores, features, mode="none")
    # passthrough preserves everything (order-insensitive)
    assert set(out.index) == set(scores.index)


# --------------------------------------------------------------------------- #
# (2) downtrend removed  /  (3) uptrend kept                                   #
# --------------------------------------------------------------------------- #
def test_soft_gate_removes_downtrend_keeps_uptrend(scores, features):
    out = apply_trend_filter(scores, features, mode="soft")
    insts_kept = set(out.index.get_level_values("instrument"))
    # On the last (fully warmed-up) date specifically:
    last_date = scores.index.get_level_values("datetime").max()
    kept_last = set(out.xs(last_date, level="datetime").index)
    assert "UP" in kept_last           # (3) uptrend kept
    assert "DOWN" not in kept_last      # (2) downtrend (falling knife) removed
    # globally the downtrend name should be heavily suppressed
    dn_rows = out.index.get_level_values("instrument") == "DOWN"
    up_rows = out.index.get_level_values("instrument") == "UP"
    assert dn_rows.sum() < up_rows.sum()


def test_strict_gate_removes_downtrend_keeps_uptrend(scores, features):
    out = apply_trend_filter(scores, features, mode="strict")
    last_date = scores.index.get_level_values("datetime").max()
    kept_last = set(out.xs(last_date, level="datetime").index)
    assert "UP" in kept_last
    assert "DOWN" not in kept_last
    # strict is at least as aggressive as soft: FLAT (no MA alignment, ~0 mom)
    # must not pass strict on the last date
    assert "FLAT" not in kept_last


def test_strict_is_subset_of_soft(scores, features):
    soft = apply_trend_filter(scores, features, mode="soft")
    strict = apply_trend_filter(scores, features, mode="strict")
    # every name passing the strict gate must also pass the soft gate
    assert set(strict.index) <= set(soft.index)


# --------------------------------------------------------------------------- #
# (4) no-lookahead                                                             #
# --------------------------------------------------------------------------- #
def test_no_lookahead_decision_uses_only_past(panel, scores):
    """Verdict at date D must be identical whether or not future rows exist.

    We pick a mid-panel date D, run the gate on the FULL panel, then truncate
    BOTH the prices and the scores to <= D, recompute, and require the D-row
    verdicts to match exactly. If the gate peeked at prices > D, truncation
    would change the answer.
    """
    dates = panel.index.get_level_values("datetime").unique().sort_values()
    d = dates[70]  # well past the 60-day warm-up, with future rows remaining

    feats_full = compute_trend_features(panel)
    out_full = apply_trend_filter(scores, feats_full, mode="strict")
    kept_full_d = set(out_full.xs(d, level="datetime").index)

    panel_trunc = panel[panel.index.get_level_values("datetime") <= d]
    scores_trunc = scores[scores.index.get_level_values("datetime") <= d]
    feats_trunc = compute_trend_features(panel_trunc)
    out_trunc = apply_trend_filter(scores_trunc, feats_trunc, mode="strict")
    kept_trunc_d = set(out_trunc.xs(d, level="datetime").index)

    assert kept_full_d == kept_trunc_d


def test_compute_features_are_trailing_not_centered(panel):
    """MA20 at row i must equal the mean of close[i-19..i] (trailing), not a
    centered/forward window."""
    feats = compute_trend_features(panel)
    up_close = panel.xs("UP", level="instrument").sort_index()
    up_feat = feats.xs("UP", level="instrument").sort_index()
    i = 40
    expected_ma20 = up_close.iloc[i - 19 : i + 1].mean()
    assert up_feat["ma20"].iloc[i] == pytest.approx(expected_ma20)


# --------------------------------------------------------------------------- #
# robustness                                                                   #
# --------------------------------------------------------------------------- #
def test_accepts_raw_close_series_directly(scores, panel):
    """If given a raw close Series (not a precomputed feature frame), the gate
    computes features internally and still satisfies the subset contract."""
    out = apply_trend_filter(scores, panel, mode="soft")
    assert set(out.index) <= set(scores.index)
    last_date = scores.index.get_level_values("datetime").max()
    kept_last = set(out.xs(last_date, level="datetime").index)
    assert "UP" in kept_last and "DOWN" not in kept_last


def test_warmup_rows_are_dropped_by_default(scores, features):
    """Rows whose trailing windows are incomplete (NaN features) cannot confirm
    a trend and must be dropped by a real gate (conservative)."""
    out = apply_trend_filter(scores, features, mode="soft")
    first_date = scores.index.get_level_values("datetime").min()
    # day 0 has no MA20 -> nothing should pass the soft gate on day 0
    assert first_date not in set(out.index.get_level_values("datetime"))


def test_empty_scores_returns_empty(features):
    empty = pd.Series(dtype=float, name="score",
                      index=pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"]))
    out = apply_trend_filter(empty, features, mode="soft")
    assert out.empty
