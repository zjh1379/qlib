from datetime import datetime

import pytest

from app.scheduling.service import is_trading_hours_cst


@pytest.mark.parametrize(
    "dt_str, expected",
    [
        ("2026-05-21 09:30", True),   # Thu, market open
        ("2026-05-21 15:00", True),   # Thu, market close
        ("2026-05-21 11:30", True),   # Thu, midday
        ("2026-05-21 08:00", False),  # Thu, before open
        ("2026-05-21 15:01", False),  # Thu, after close
        ("2026-05-23 11:00", False),  # Sat
        ("2026-05-24 22:00", False),  # Sun
    ],
)
def test_trading_hours_cst(dt_str, expected):
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    assert is_trading_hours_cst(dt) == expected
