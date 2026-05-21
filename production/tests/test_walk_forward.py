from datetime import date, timedelta

import pytest

from production.walk_forward import HorizonConfig, WalkForwardSplit, _LABEL_BUFFER_DAYS, split


# Per-spec table — 1d/5d/20d horizons with different train lengths
CFG_1D = HorizonConfig(name="1d", train_years=3, valid_years=1, stack_years=1, test_weeks=1)
CFG_5D = HorizonConfig(name="5d", train_years=5, valid_years=1, stack_years=1, test_weeks=1)
CFG_20D = HorizonConfig(name="20d", train_years=7, valid_years=1, stack_years=1, test_weeks=1)


@pytest.mark.parametrize("cfg", [CFG_1D, CFG_5D, CFG_20D])
def test_split_returns_four_windows_per_horizon(cfg):
    s = split(end_date=date(2026, 5, 17), cfg=cfg)
    assert isinstance(s, WalkForwardSplit)
    assert s.train_start < s.train_end
    assert s.valid_start < s.valid_end
    assert s.stack_start < s.stack_end
    assert s.test_start < s.test_end


def test_test_window_ends_on_end_date():
    s = split(end_date=date(2026, 5, 17), cfg=CFG_5D)
    assert s.test_end == date(2026, 5, 17)
    assert (s.test_end - s.test_start).days == 7 - 1  # inclusive 7-day window


def test_no_overlap_train_valid_stack_test():
    """CRITICAL — walk-forward off-by-one is risk R8 in the spec."""
    for cfg in (CFG_1D, CFG_5D, CFG_20D):
        s = split(end_date=date(2026, 5, 17), cfg=cfg)
        # Each window's start > previous window's end
        assert s.train_end < s.valid_start, f"{cfg.name}: train and valid overlap"
        assert s.valid_end < s.stack_start, f"{cfg.name}: valid and stack overlap"
        assert s.stack_end < s.test_start, f"{cfg.name}: stack and test overlap"

        # And gap between train_end and valid_start is exactly 1 day (no leak room)
        assert (s.valid_start - s.train_end).days == 1, f"{cfg.name}: train→valid gap"
        assert (s.stack_start - s.valid_end).days == 1, f"{cfg.name}: valid→stack gap"
        assert (s.test_start - s.stack_end).days == 1, f"{cfg.name}: stack→test gap"


def test_horizons_have_different_train_starts():
    s1 = split(end_date=date(2026, 5, 17), cfg=CFG_1D)
    s5 = split(end_date=date(2026, 5, 17), cfg=CFG_5D)
    s20 = split(end_date=date(2026, 5, 17), cfg=CFG_20D)
    # 1d → 3y train, 5d → 5y, 20d → 7y. Train starts increase in lookback.
    assert s1.train_start > s5.train_start
    assert s5.train_start > s20.train_start


def test_label_horizon_subtraction():
    """The training window must end ≥ horizon days before valid_start to allow
    realized labels to materialize for each training sample."""
    for cfg in (CFG_1D, CFG_5D, CFG_20D):
        s = split(end_date=date(2026, 5, 17), cfg=cfg)
        expected_buffer = _LABEL_BUFFER_DAYS[cfg.name]
        assert (s.valid_start - s.train_end).days >= 1
        assert s.train_label_end == s.train_end - timedelta(days=expected_buffer), (
            f"{cfg.name}: train_label_end mismatch"
        )
