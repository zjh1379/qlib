import pytest

import numpy as np
import pandas as pd

from production.consensus import consensus_score, write_pred_pkl


def test_consensus_all_positive_is_1():
    preds = np.array([0.1, 0.2, 0.3])
    assert consensus_score(preds) == 1.0


def test_consensus_all_negative_is_1():
    preds = np.array([-0.1, -0.2, -0.3])
    assert consensus_score(preds) == 1.0


def test_consensus_balanced_is_low():
    preds = np.array([0.1, -0.1, 0.0])  # sign(0.0)=0 -> counts as 0
    # |sum(signs)| = |1 + -1 + 0| = 0
    assert consensus_score(preds) == 0.0


def test_consensus_5_of_9_positive_is_5_over_9():
    preds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4])
    # |sum(signs)| / 9 = |5 - 4| / 9 = 1/9
    assert consensus_score(preds) == pytest.approx(1 / 9)


def test_consensus_per_row_handles_nan_columns():
    """Rows with NaN entries get a fractional consensus over the non-NaN columns."""
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-05-15"), "SH600000"),
            (pd.Timestamp("2026-05-15"), "SH600001"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {
            "lgbm_1d": [0.1, 0.2],
            "lgbm_5d": [-0.1, np.nan],
            "lgbm_20d": [np.nan, 0.3],
        },
        index=idx,
    )
    from production.consensus import consensus_per_row
    out = consensus_per_row(df)
    # SH600000: signs = [+1, -1, nan] -> 1 of 2 valid → |0| / 2 = 0.0
    # SH600001: signs = [+1, nan, +1] -> 2 of 2 valid → |2| / 2 = 1.0
    assert out.loc[(pd.Timestamp("2026-05-15"), "SH600000")] == pytest.approx(0.0)
    assert out.loc[(pd.Timestamp("2026-05-15"), "SH600001")] == pytest.approx(1.0)


def test_consensus_per_row_all_nan_row_returns_nan():
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-05-15"), "SH600000")],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {"lgbm_1d": [np.nan], "lgbm_5d": [np.nan]},
        index=idx,
    )
    from production.consensus import consensus_per_row
    out = consensus_per_row(df)
    assert pd.isna(out.iloc[0])


def test_write_pred_pkl_roundtrip(tmp_path):
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-05-15"), "SH600000"),
            (pd.Timestamp("2026-05-15"), "SH600001"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {
            "score": [0.10, -0.05],
            "consensus": [1.0, 0.33],
            "lgbm_1d": [0.08, -0.04],
            "lgbm_5d": [0.11, -0.05],
            "lgbm_20d": [0.12, -0.06],
        },
        index=idx,
    )
    out_path = tmp_path / "pred.pkl"
    write_pred_pkl(df, out_path)
    loaded = pd.read_pickle(out_path)
    pd.testing.assert_frame_equal(loaded, df)
