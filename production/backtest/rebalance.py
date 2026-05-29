"""Rebalance policies. Each returns equal-weighted long-only target weights."""
from __future__ import annotations

import pandas as pd


def _equal_top_k(scores: pd.Series, k: int) -> pd.Series:
    s = scores.dropna()
    if s.empty or k <= 0:
        return pd.Series(dtype=float)
    top = s.nlargest(k).index
    return pd.Series(1.0 / len(top), index=top)


class RebalancePolicy:
    def should_rebalance(self, step: int) -> bool:
        raise NotImplementedError

    def target_weights(self, scores: pd.Series, current: pd.Series) -> pd.Series:
        raise NotImplementedError


class Daily(RebalancePolicy):
    def __init__(self, top_k: int = 30):
        self.top_k = top_k

    def should_rebalance(self, step: int) -> bool:
        return True

    def target_weights(self, scores: pd.Series, current: pd.Series) -> pd.Series:
        return _equal_top_k(scores, self.top_k)
