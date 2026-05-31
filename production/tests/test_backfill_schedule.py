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


# ---------------------------------------------------------------------------
# T2: fold_recorders_complete — skip-existing predicate
# ---------------------------------------------------------------------------

from production.rolling_train import fold_recorders_complete  # noqa: E402


class _FakeExp:
    def __init__(self, names):
        self._names = names

    def list_recorders(self):
        return [type("R", (), {"info": {"name": n}})() for n in self._names]


def test_fold_complete_true_when_all_present():
    names = [f"{m}_{h}_2026-01-02" for m in ("lgbm", "alstm", "tra") for h in ("1d", "5d", "20d")]
    assert fold_recorders_complete(
        _FakeExp(names), date(2026, 1, 2),
        ("lgbm", "alstm", "tra"), ("1d", "5d", "20d"),
    ) is True


def test_fold_complete_false_when_missing_one():
    names = [f"lgbm_{h}_2026-01-02" for h in ("1d", "5d", "20d")]  # only lgbm
    assert fold_recorders_complete(
        _FakeExp(names), date(2026, 1, 2),
        ("lgbm", "alstm", "tra"), ("1d", "5d", "20d"),
    ) is False
