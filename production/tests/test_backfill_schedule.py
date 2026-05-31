"""Tests for backfill schedule helpers (T1 + T2).

T1: backfill_fold_end_dates — pure schedule function
T2: fold_recorders_complete — skip-existing predicate
"""
from datetime import date

from production.rolling_train import backfill_fold_end_dates


def test_weekly_step_default():
    # 2021-01-01 is a Friday-anchored walk; step_weeks=1 -> weekly
    out = backfill_fold_end_dates(date(2021, 1, 1), date(2021, 2, 1), step_weeks=1)
    assert out[0].weekday() == 4          # all Fridays
    assert (out[1] - out[0]).days == 7
    assert all(d.weekday() == 4 for d in out)


def test_semiannual_step_contiguous():
    out = backfill_fold_end_dates(date(2021, 1, 1), date(2026, 1, 1), step_weeks=26)
    assert (out[1] - out[0]).days == 26 * 7
    assert 8 <= len(out) <= 12            # ~10 folds over 5y


def test_annual_step():
    out = backfill_fold_end_dates(date(2021, 1, 1), date(2026, 1, 1), step_weeks=52)
    assert (out[1] - out[0]).days == 52 * 7
    assert 4 <= len(out) <= 6             # ~5 folds
