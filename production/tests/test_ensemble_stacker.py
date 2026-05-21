import numpy as np
import pandas as pd
import pytest

from production.ensemble_stacker import RidgeStacker


def _mk_oof(n_days=20, n_stocks=30, n_bases=9, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_days)
    stocks = [f"SH60{i:04d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    cols = [f"base_{i}" for i in range(n_bases)]
    target = rng.normal(0, 0.02, size=len(idx))
    base_preds = pd.DataFrame(
        {c: target + rng.normal(0, 0.01, size=len(idx)) for c in cols},
        index=idx,
    )
    y = pd.Series(target, index=idx, name="label")
    return base_preds, y


def test_stacker_fits_and_predicts():
    base_preds, y = _mk_oof()
    stacker = RidgeStacker()
    stacker.fit_oof(base_preds, y)
    test_base, _ = _mk_oof(seed=1)
    out = stacker.predict(test_base)
    assert isinstance(out, pd.Series)
    assert out.shape == (len(test_base),)


def test_stacker_grid_searches_alpha():
    base_preds, y = _mk_oof()
    stacker = RidgeStacker(alpha_grid=[0.1, 1.0, 10.0])
    stacker.fit_oof(base_preds, y)
    assert stacker.alpha in (0.1, 1.0, 10.0)


def test_stacker_z_scores_cross_sectionally():
    """Stacker inputs must be cross-sectionally z-scored per day."""
    base_preds, y = _mk_oof()
    stacker = RidgeStacker()
    z = stacker._cross_sectional_zscore(base_preds)
    for d in z.index.get_level_values("datetime").unique()[:3]:
        slice_ = z.xs(d, level="datetime")
        assert slice_.mean().abs().max() < 1e-6
        assert (slice_.std(ddof=0) - 1).abs().max() < 1e-6


def test_stacker_fallback_to_rank_average_when_fit_fails():
    """If Ridge fails (e.g. singular matrix), fall back to rank_average."""
    base_preds, _ = _mk_oof(n_days=1, n_stocks=2, n_bases=9)
    stacker = RidgeStacker()
    try:
        stacker.fit_oof(base_preds, pd.Series(dtype="float64"))
    except Exception:
        pass
    out = stacker.predict_with_fallback(base_preds)
    assert out is not None
