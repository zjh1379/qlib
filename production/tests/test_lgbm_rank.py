"""TDD for production/lgbm_rank.py — LambdaRank LGBM for stock ranking.

The crux is converting continuous forward returns into LightGBM lambdarank
inputs: per-DAY integer relevance grades (best return that day -> highest grade)
+ per-day query GROUP counts (lambdarank needs data sorted by group, with the
size of each group). These tests pin that contract offline (no lightgbm/qlib).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from production.lgbm_rank import relevance_grades_and_groups


def _y(series_by_day: dict[str, list[tuple[str, float]]]) -> pd.Series:
    rows, idx = [], []
    for day, pairs in series_by_day.items():
        for inst, val in pairs:
            idx.append((pd.Timestamp(day), inst))
            rows.append(val)
    mi = pd.MultiIndex.from_tuples(idx, names=["datetime", "instrument"])
    return pd.Series(rows, index=mi, name="label")


def test_groups_sum_to_len_and_sorted_by_day():
    y = _y({
        "2024-01-02": [("A", 0.1), ("B", -0.2), ("C", 0.05)],
        "2024-01-01": [("D", 0.3), ("E", 0.0)],
    })
    grades, groups, idx = relevance_grades_and_groups(y, n_grades=4)
    # groups are per-day counts in chronological order: 2024-01-01 (2), then 01-02 (3)
    assert list(groups) == [2, 3]
    assert int(np.sum(groups)) == len(y) == len(grades) == len(idx)
    # returned index is sorted by datetime ascending
    days = idx.get_level_values("datetime")
    assert list(days) == sorted(days)


def test_best_return_gets_top_grade_each_day():
    y = _y({
        "2024-01-01": [("A", 0.10), ("B", 0.20), ("C", -0.10), ("D", 0.05)],
    })
    grades, groups, idx = relevance_grades_and_groups(y, n_grades=4)
    g = pd.Series(grades, index=idx).xs(pd.Timestamp("2024-01-01"), level="datetime")
    assert g["B"] == 3            # best return -> highest grade
    assert g["C"] == 0            # worst return -> lowest grade
    assert g["B"] > g["A"] > g["C"]


def test_grades_within_range_and_integer():
    rng = np.random.default_rng(0)
    y = _y({f"2024-01-{d:02d}": [(f"S{i}", float(rng.normal())) for i in range(50)]
            for d in range(1, 8)})
    grades, groups, idx = relevance_grades_and_groups(y, n_grades=16)
    assert grades.dtype.kind in "iu"
    assert grades.min() >= 0 and grades.max() <= 15


def test_handles_ties_and_constant_day():
    y = _y({"2024-01-01": [("A", 0.0), ("B", 0.0), ("C", 0.0)]})
    grades, groups, idx = relevance_grades_and_groups(y, n_grades=8)
    assert len(grades) == 3 and list(groups) == [3]
    assert grades.min() >= 0 and grades.max() <= 7   # no crash, valid range


def test_grades_align_with_returned_index():
    y = _y({
        "2024-01-01": [("A", 0.1), ("B", 0.2)],
        "2024-01-02": [("C", -0.1), ("D", 0.4)],
    })
    grades, groups, idx = relevance_grades_and_groups(y, n_grades=4)
    s = pd.Series(grades, index=idx)
    # D had the best (only-positive-largest) return on its day -> top grade that day
    assert s.xs(pd.Timestamp("2024-01-02"), level="datetime")["D"] == 3
