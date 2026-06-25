"""Tests for production/safety_watchdog.py hardening (P0.1–P0.4)."""
from production.safety_watchdog import decide_action, is_killable_cmd


def test_decide_action_ok_when_low():
    assert decide_action(50.0, 18.0, 80.0, warn_pct=80.0, kill_pct=92.0, floor_gb=4.0) == "ok"


def test_decide_action_warn_at_warn_pct():
    assert decide_action(85.0, 68.0, 80.0, warn_pct=80.0, kill_pct=92.0, floor_gb=4.0) == "warn"


def test_decide_action_kill_at_kill_pct():
    assert decide_action(93.0, 74.0, 80.0, warn_pct=80.0, kill_pct=92.0, floor_gb=4.0) == "kill"


def test_decide_action_kill_on_absolute_floor_even_if_pct_low():
    # free = 80 - 77 = 3GB < 4GB floor -> kill regardless of pct
    assert decide_action(70.0, 77.0, 80.0, warn_pct=80.0, kill_pct=92.0, floor_gb=4.0) == "kill"


# ---------------------------------------------------------------------------
# P0.2 — is_killable_cmd
# ---------------------------------------------------------------------------

def test_is_killable_matches_training_tokens():
    assert is_killable_cmd("python -m production.train_alstm --end-date 2026-06-20")
    assert is_killable_cmd("python -m production.rolling_train run-once")
    assert is_killable_cmd("python -m production.run_split --end-date 2026-06-20")
    assert is_killable_cmd("python -m production.train_tra ...")
    assert is_killable_cmd("python -m production.walk_forward ...")


def test_is_killable_protects_infra():
    assert not is_killable_cmd("uvicorn app.main:app --port 8000")
    assert not is_killable_cmd("node vite")
    assert not is_killable_cmd("chrome.exe --type=renderer")
