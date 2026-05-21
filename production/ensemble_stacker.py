"""Ridge stacking meta-learner with OOF training.

Trained on per-day cross-sectionally z-scored base preds -> realized
open-to-open return. Hyperparameter: alpha selected per-week via 3-point
grid search on the provided validation tail of OOF data.

Fallback chain:
  1. RidgeStacker.predict
  2. rank_average over available base columns
  3. (handled upstream) roll back to last week's recorder
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from production.ensemble_rank_avg import rank_average

_log = logging.getLogger("ensemble_stacker")


class RidgeStacker:
    def __init__(self, alpha_grid: Iterable[float] = (0.1, 1.0, 10.0)):
        self.alpha_grid = list(alpha_grid)
        self.alpha: float | None = None
        self.coefficients_: pd.Series | None = None
        self.intercept_: float | None = None
        self._fit_columns: list[str] | None = None

    @staticmethod
    def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
        def _z(s: pd.Series) -> pd.Series:
            mu = s.mean()
            sd = s.std(ddof=0)
            if sd == 0 or pd.isna(sd):
                return s - mu
            return (s - mu) / sd

        return df.groupby(level="datetime").transform(_z)

    def fit_oof(self, base_preds: pd.DataFrame, y: pd.Series) -> "RidgeStacker":
        if base_preds.empty or y.empty:
            raise ValueError("empty OOF training inputs")

        joined = base_preds.join(y.rename("__y__"), how="inner").dropna()
        if joined.empty:
            raise ValueError("no overlapping (date, instrument) rows between base_preds and y")

        X_raw = joined[[c for c in joined.columns if c != "__y__"]]
        X = self._cross_sectional_zscore(X_raw).fillna(0.0)
        y_aligned = joined["__y__"]
        self._fit_columns = list(X.columns)

        # Grid search alpha by validation IC on the held-out last 20% of dates
        dates = sorted(X.index.get_level_values("datetime").unique())
        cut = int(len(dates) * 0.8)
        train_dates, val_dates = set(dates[:cut]), set(dates[cut:])
        X_train = X[X.index.get_level_values("datetime").isin(train_dates)]
        y_train = y_aligned[y_aligned.index.get_level_values("datetime").isin(train_dates)]
        X_val = X[X.index.get_level_values("datetime").isin(val_dates)]
        y_val = y_aligned[y_aligned.index.get_level_values("datetime").isin(val_dates)]

        best_alpha, best_ic = None, -np.inf
        for a in self.alpha_grid:
            mdl = Ridge(alpha=a)
            mdl.fit(X_train.to_numpy(), y_train.to_numpy())
            pred_val = pd.Series(mdl.predict(X_val.to_numpy()), index=X_val.index)
            df_eval = pd.DataFrame({"pred": pred_val, "y": y_val}).dropna()
            ics = df_eval.groupby(level="datetime").apply(
                lambda g: g["pred"].corr(g["y"]) if len(g) > 2 else np.nan
            ).dropna()
            ic_mean = ics.mean() if len(ics) else -np.inf
            if ic_mean > best_ic:
                best_alpha, best_ic = a, ic_mean

        self.alpha = best_alpha if best_alpha is not None else self.alpha_grid[0]

        final = Ridge(alpha=self.alpha)
        final.fit(X.to_numpy(), y_aligned.to_numpy())
        self.coefficients_ = pd.Series(final.coef_, index=self._fit_columns)
        self.intercept_ = float(final.intercept_)
        _log.info(
            "stacker_fit alpha=%s best_val_ic=%s coefs=%s",
            self.alpha, best_ic, self.coefficients_.to_dict(),
        )
        return self

    def predict(self, base_preds: pd.DataFrame) -> pd.Series:
        if self.coefficients_ is None or self._fit_columns is None:
            raise RuntimeError("Stacker must be fit before predict")
        X_raw = base_preds.reindex(columns=self._fit_columns)
        X = self._cross_sectional_zscore(X_raw).fillna(0.0)
        out = X.to_numpy() @ self.coefficients_.to_numpy() + (self.intercept_ or 0.0)
        return pd.Series(out, index=X.index, name="score_stacked")

    def predict_with_fallback(self, base_preds: pd.DataFrame) -> pd.Series:
        """Try Ridge first; fall back to rank_average if anything fails."""
        try:
            return self.predict(base_preds)
        except Exception as exc:
            _log.warning("stacker_predict_failed_falling_back_to_rank_avg error=%s", str(exc))
            ranks = rank_average(base_preds)
            return (-ranks).rename("score_rank_avg_fallback")
