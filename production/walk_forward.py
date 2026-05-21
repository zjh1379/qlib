"""Walk-forward splitter for the rolling retrain pipeline.

Per spec Section 6, each horizon has its own (train, valid, stack-fit, test)
windows that slide forward together by 7 days every week.

Invariant tested in test_walk_forward.test_no_overlap_train_valid_stack_test.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class HorizonConfig:
    name: str  # "1d", "5d", "20d"
    train_years: int
    valid_years: int
    stack_years: int
    test_weeks: int


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    stack_start: date
    stack_end: date
    test_start: date
    test_end: date
    # train_label_end accounts for the N-day label horizon — training samples
    # past this date can't have their realized return computed yet.
    train_label_end: date
    horizon: str


# Trading-day buffer per horizon (calendar days). 5d label ~ 7 calendar buffer.
_LABEL_BUFFER_DAYS = {"1d": 3, "5d": 7, "20d": 30}


def _years_ago(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:  # Feb 29 -> Feb 28
        return d.replace(year=d.year - years, day=28)


def split(end_date: date, cfg: HorizonConfig) -> WalkForwardSplit:
    """Build a walk-forward split anchored at `end_date` (the last day of the test window)."""
    test_end = end_date
    test_start = test_end - timedelta(days=cfg.test_weeks * 7 - 1)

    stack_end = test_start - timedelta(days=1)
    stack_start = _years_ago(stack_end, cfg.stack_years) + timedelta(days=1)

    valid_end = stack_start - timedelta(days=1)
    valid_start = _years_ago(valid_end, cfg.valid_years) + timedelta(days=1)

    train_end = valid_start - timedelta(days=1)
    train_start = _years_ago(train_end, cfg.train_years) + timedelta(days=1)

    train_label_end = train_end - timedelta(days=_LABEL_BUFFER_DAYS[cfg.name])

    return WalkForwardSplit(
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        stack_start=stack_start,
        stack_end=stack_end,
        test_start=test_start,
        test_end=test_end,
        train_label_end=train_label_end,
        horizon=cfg.name,
    )
