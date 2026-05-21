"""Equal-weight rank-average ensemble.

Used as:
  (a) the simplest baseline ensemble (Phase D, 2-model);
  (b) the fallback when the Ridge stacker fails (Phase E).

Higher score -> better -> rank 1 (we invert to keep "lower = better").
"""
from __future__ import annotations

import pandas as pd


def rank_average(base_preds: pd.DataFrame) -> pd.Series:
    """Return per-row average cross-sectional rank across available base columns.

    Lower returned value = stronger predicted alpha (consistent with rank_avg in
    backend/app/models/schemas.py).
    """
    # Per-day per-column descending rank -> 1 is highest score
    ranks = base_preds.groupby(level="datetime").rank(ascending=False, method="min")
    return ranks.mean(axis=1, skipna=True).rename("score_rank_avg")
