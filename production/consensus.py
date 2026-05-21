"""Consensus score + unified pred.pkl writer.

`consensus` = fraction of base predictions agreeing in direction, in [0, 1].
A score of 1.0 means all base models agreed on sign.

The unified pred.pkl schema (consumed by backend/app/models/service.py):
    Index: MultiIndex(datetime, instrument)
    Columns:
        score:       the post-processed ensemble score (raw scalar)
        consensus:   in [0, 1]
        base_scores: this is *expanded* into one column per base model output,
                     e.g. lgbm_1d, lgbm_5d, lgbm_20d, alstm_1d, ...
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def consensus_score(base_preds: np.ndarray) -> float:
    """Return |sum sign(p_i)| / N for an array of base predictions."""
    signs = np.sign(base_preds)
    return float(abs(signs.sum()) / len(base_preds))


def consensus_per_row(base_preds_df: pd.DataFrame) -> pd.Series:
    """Vectorized version: returns a Series of consensus scores per row.

    NaN-safe: rows with partial coverage (some columns NaN) get a fractional
    score based on the non-NaN entries. Rows with all NaN get NaN.
    """
    signs = np.sign(base_preds_df.to_numpy())
    valid_count = np.sum(~np.isnan(signs), axis=1)
    summed = np.abs(np.nansum(signs, axis=1))
    out = np.where(valid_count > 0, summed / np.where(valid_count > 0, valid_count, 1), np.nan)
    return pd.Series(out, index=base_preds_df.index, name="consensus")


def write_pred_pkl(df: pd.DataFrame, path: Path) -> None:
    """Persist the unified prediction frame to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
