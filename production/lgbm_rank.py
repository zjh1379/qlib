"""LambdaRank LGBM for cross-sectional stock ranking.

qlib's stock `LGBModel` only supports loss in {mse, binary} and builds
`lgb.Dataset` WITHOUT a group array, so it can't do learning-to-rank. This
module adds `LGBRankModel`, a drop-in qlib Model that trains LightGBM with
`objective="lambdarank"`:

  - converts the continuous forward-return label into per-DAY integer relevance
    grades (best return that day -> highest grade), and
  - passes per-day query GROUP sizes (lambdarank requires the data sorted by
    group, with each group's size),

so the gradient concentrates on getting the TOP of each day's ranking right —
exactly where this strategy's alpha lives (see 2026-06-04-robustness-results.md:
the edge is concentrated in the top ~5 names).

`predict` returns a score Series (higher = better), identical in shape to
`LGBModel.predict`, so pooling / ensemble / daily_inference work unchanged.
"""
from __future__ import annotations

import sys as _sys
import sysconfig as _sysconfig
_P = _sysconfig.get_paths().get("purelib")
if _P and _P not in _sys.path[:1]:
    _sys.path.insert(0, _P)

import logging

import numpy as np
import pandas as pd

_log = logging.getLogger("lgbm_rank")


def relevance_grades_and_groups(y: pd.Series, n_grades: int = 16):
    """Continuous forward returns -> (grades, group_sizes, sorted_index) for
    LightGBM lambdarank.

    - drops NaN labels, sorts datetime-major (so query groups are contiguous),
    - grade = per-day cross-sectional percentile rank bucketed into 0..n_grades-1
      (best return that day -> n_grades-1, worst -> 0),
    - group_sizes = number of names per day, in chronological order (sum == len).

    Returns (grades:int ndarray, group_sizes:list[int], index:MultiIndex) all
    aligned to the same datetime-major ordering.
    """
    s = y.dropna()
    names = list(s.index.names)
    if names and names[0] != "datetime":
        # ensure datetime is the outer (major) level so groups are contiguous
        s = s.swaplevel().sort_index()
    else:
        s = s.sort_index()
    if s.empty:
        return np.empty(0, dtype=int), [], s.index

    pct = s.groupby(level="datetime").rank(method="average", pct=True)  # (0,1], higher=better
    grades = np.clip((pct.values * n_grades - 1e-9).astype(int), 0, n_grades - 1)
    group_sizes = s.groupby(level="datetime", sort=True).size().tolist()
    return grades.astype(int), group_sizes, s.index


class LGBRankModel:
    """qlib-compatible (duck-typed fit/predict) LightGBM LambdaRank model.

    Accepts the same hyperparam kwargs as the mse LGBModel config (it ignores
    `loss`), plus `n_grades` and `eval_at`. Picklable (holds an lgb.Booster).
    """

    def __init__(self, n_grades: int = 16, eval_at=(5, 10),
                 num_boost_round: int = 1000, early_stopping_rounds: int = 50,
                 **kwargs):
        kwargs.pop("loss", None)  # mse-config compatibility; objective is fixed below
        self.n_grades = int(n_grades)
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": list(eval_at),
            "label_gain": [float(2 ** i - 1) for i in range(self.n_grades)],
            "verbosity": -1,
        }
        self.params.update(kwargs)
        self.model = None

    # ---- qlib Model interface ------------------------------------------------
    def fit(self, dataset, **kwargs):
        import lightgbm as lgb
        from qlib.data.dataset.handler import DataHandlerLP
        df_tr, df_va = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        def _xy(df):
            x = df["feature"]
            y = df["label"]
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            return x, y

        x_tr, y_tr = _xy(df_tr)
        x_va, y_va = _xy(df_va)
        g_tr, grp_tr, idx_tr = relevance_grades_and_groups(y_tr, self.n_grades)
        g_va, grp_va, idx_va = relevance_grades_and_groups(y_va, self.n_grades)
        if len(idx_tr) == 0:
            raise ValueError("LGBRankModel.fit: empty training data after dropna")
        # align features to the datetime-major (sorted, non-nan) order
        x_tr = x_tr.loc[idx_tr]
        x_va = x_va.loc[idx_va]

        dtrain = lgb.Dataset(x_tr.values, label=g_tr, group=grp_tr)
        dvalid = lgb.Dataset(x_va.values, label=g_va, group=grp_va, reference=dtrain)
        self.model = lgb.train(
            self.params, dtrain, num_boost_round=self.num_boost_round,
            valid_sets=[dtrain, dvalid], valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds),
                       lgb.log_evaluation(50)],
        )
        return self

    def predict(self, dataset, segment="test"):
        from qlib.data.dataset.handler import DataHandlerLP
        if self.model is None:
            raise ValueError("LGBRankModel is not fitted yet!")
        x = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        if x is None or len(x) == 0:
            return pd.Series(dtype=float)
        return pd.Series(self.model.predict(x.values), index=x.index)
