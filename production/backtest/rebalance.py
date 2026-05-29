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


class FixedPeriod(RebalancePolicy):
    def __init__(self, top_k: int = 30, period: int = 5):
        self.top_k = top_k
        self.period = max(1, period)

    def should_rebalance(self, step: int) -> bool:
        return step % self.period == 0

    def target_weights(self, scores: pd.Series, current: pd.Series) -> pd.Series:
        return _equal_top_k(scores, self.top_k)


class Banded(RebalancePolicy):
    """Hysteresis: buy a name when rank <= top_k; only sell a held name when
    its rank drops beyond exit_k. Naturally extends holding + cuts turnover."""

    def __init__(self, top_k: int = 30, exit_k: int | None = None):
        self.top_k = top_k
        self.exit_k = exit_k if exit_k is not None else 2 * top_k

    def should_rebalance(self, step: int) -> bool:
        return True

    def target_weights(self, scores: pd.Series, current: pd.Series) -> pd.Series:
        s = scores.dropna()
        if s.empty:
            return pd.Series(dtype=float)
        ranks = s.rank(ascending=False, method="first")
        held = list(current[current > 0].index) if current is not None and not current.empty else []
        keep = [i for i in held if i in ranks.index and ranks[i] <= self.exit_k]
        need = self.top_k - len(keep)
        if need > 0:
            cand = ranks[ranks <= self.top_k].sort_values().index
            for i in cand:
                if i not in keep:
                    keep.append(i)
                    need -= 1
                    if need == 0:
                        break
        keep = keep[: self.top_k]
        if not keep:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(keep), index=pd.Index(keep))
