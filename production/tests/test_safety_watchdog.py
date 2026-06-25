"""Tests for production/safety_watchdog.py hardening (P0.1–P0.4)."""
from production.safety_watchdog import decide_action


def test_decide_action_ok_when_low():
    assert decide_action(50.0, 18.0, 80.0, warn_pct=80.0, kill_pct=92.0, floor_gb=4.0) == "ok"


def test_decide_action_warn_at_warn_pct():
    assert decide_action(85.0, 68.0, 80.0, warn_pct=80.0, kill_pct=92.0, floor_gb=4.0) == "warn"


def test_decide_action_kill_at_kill_pct():
    assert decide_action(93.0, 74.0, 80.0, warn_pct=80.0, kill_pct=92.0, floor_gb=4.0) == "kill"


def test_decide_action_kill_on_absolute_floor_even_if_pct_low():
    # free = 80 - 77 = 3GB < 4GB floor -> kill regardless of pct
    assert decide_action(70.0, 77.0, 80.0, warn_pct=80.0, kill_pct=92.0, floor_gb=4.0) == "kill"
