"""Isotonic regression calibration for model scores -> expected returns.

Per the design spec, each horizon (1d/5d/20d) has a separate isotonic map
fitted on the validation slice's (composite_score, realized_return) pairs.

composite_score := -rank_avg over that horizon's model columns.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

log = logging.getLogger(__name__)

MIN_SAMPLES = 100


def _composite_score(pred_df: pd.DataFrame, horizon: str) -> pd.Series:
    """Compute -rank_avg of all <model>_<horizon> columns per datetime."""
    cols = [c for c in pred_df.columns if c.endswith(f"_{horizon}")]
    if not cols:
        return pd.Series(dtype=float, index=pred_df.index)
    ranks = pred_df[cols].groupby(level="datetime").rank(ascending=False, method="min")
    return -ranks.mean(axis=1, skipna=True)


def fit_calibration(
    pred_df: pd.DataFrame,
    label_df: pd.DataFrame,
    horizons: Iterable[str] = ("1d", "5d", "20d"),
) -> dict[str, IsotonicRegression]:
    """Fit one IsotonicRegression per horizon.

    pred_df: MultiIndex (datetime, instrument), columns include <model>_<horizon>
    label_df: same MultiIndex, columns include label_<horizon>

    Returns dict horizon -> fitted IsotonicRegression. Horizons with fewer
    than MIN_SAMPLES non-NaN observations are silently skipped (logged).
    """
    out: dict[str, IsotonicRegression] = {}
    for h in horizons:
        label_col = f"label_{h}"
        if label_col not in label_df.columns:
            log.warning("calibration_skip horizon=%s reason=label_missing", h)
            continue
        x = _composite_score(pred_df, h)
        y = label_df[label_col]
        if x.empty:
            log.warning("calibration_skip horizon=%s reason=no_pred_cols", h)
            continue
        df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if len(df) < MIN_SAMPLES:
            log.warning(
                "calibration_skip horizon=%s samples=%d threshold=%d",
                h, len(df), MIN_SAMPLES,
            )
            continue
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(df["x"].values, df["y"].values)
        out[h] = iso
        log.info("calibration_fit horizon=%s samples=%d", h, len(df))
    return out


def apply_calibration(
    composite_scores: pd.Series,
    iso: IsotonicRegression,
) -> pd.Series:
    """Apply fitted isotonic to composite scores. NaN inputs -> NaN output.

    The isotonic regression is fitted with out_of_bounds='clip' so very
    large or very small inputs are mapped to the boundary of the fitted
    range rather than extrapolated.
    """
    if composite_scores.empty:
        return composite_scores
    mask = composite_scores.notna()
    out = pd.Series(np.nan, index=composite_scores.index, dtype=float)
    if mask.any():
        out.loc[mask] = iso.predict(composite_scores.loc[mask].values)
    return out


def save_calibration(
    cal: dict[str, IsotonicRegression],
    path: Path | str,
    meta: dict | None = None,
) -> None:
    payload = {"maps": cal, "meta": meta or {}}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(payload, f)


def load_calibration(path: Path | str) -> dict:
    """Returns {'maps': {'1d': iso, ...}, 'meta': {...}} or empty if missing."""
    p = Path(path)
    if not p.exists():
        return {"maps": {}, "meta": {}}
    with p.open("rb") as f:
        return pickle.load(f)
