from datetime import date, timedelta

import pandas as pd
import pytest

from production.shadow_tracker import ShadowTracker


def test_shadow_starts_tracking_on_first_run(tmp_path):
    t = ShadowTracker(state_path=tmp_path / "shadow.json")
    t.record_run(recorder_id="abc123", run_date=date(2026, 5, 17), is_shadow=True)
    state = t.get_state("abc123")
    assert state["weeks_observed"] == 1


def test_shadow_promotes_after_4_weeks_if_better(tmp_path):
    t = ShadowTracker(state_path=tmp_path / "shadow.json")
    for i in range(4):
        d = date(2026, 5, 17) + timedelta(weeks=i)
        t.record_run(recorder_id="abc123", run_date=d, is_shadow=True, ir=2.8)
        t.record_baseline(recorder_id="prod_xyz", run_date=d, ir=2.2)
    decision = t.evaluate_promotion("abc123")
    assert decision["promote"] is True
    assert decision["ir_delta"] == pytest.approx(0.6)


def test_shadow_does_not_promote_before_4_weeks(tmp_path):
    t = ShadowTracker(state_path=tmp_path / "shadow.json")
    for i in range(2):
        d = date(2026, 5, 17) + timedelta(weeks=i)
        t.record_run(recorder_id="abc123", run_date=d, is_shadow=True, ir=3.0)
        t.record_baseline(recorder_id="prod", run_date=d, ir=2.0)
    decision = t.evaluate_promotion("abc123")
    assert decision["promote"] is False
    assert decision["reason"] == "insufficient_weeks"
