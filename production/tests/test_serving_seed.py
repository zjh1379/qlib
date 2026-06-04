"""Tests for the serving-recorder seed guard (train_helpers).

Root-cause regression test: the automated pooling paths
(rolling_train.run_once + run_split._pool_from_recorders) must seed pred.pkl
into the ensemble_<date> recorder so the backend (get_latest_recorder_id) and
daily_inference can serve/extend it — BUT only for live (recent) folds. A
historical backfill fold must NOT seed serving, because get_latest_recorder_id
and _find_pooled_recorder sort by recorder start_time, so a backfill's
last-trained (newest start_time) fold would otherwise hijack serving with a
stale DATA date.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from production.train_helpers import is_live_fold, seed_serving_recorder


# --------------------------------------------------------------------------- #
# is_live_fold — the recency predicate                                         #
# --------------------------------------------------------------------------- #
def test_is_live_fold_recent_true():
    today = date(2026, 6, 7)
    assert is_live_fold(date(2026, 6, 5), today=today) is True   # 2 days
    assert is_live_fold(date(2026, 6, 7), today=today) is True   # same day


def test_is_live_fold_old_false():
    today = date(2026, 6, 7)
    assert is_live_fold(date(2025, 12, 26), today=today) is False  # backfill fold
    assert is_live_fold(date(2026, 5, 1), today=today) is False


def test_is_live_fold_boundary():
    today = date(2026, 6, 11)
    # default max_age_days = 10
    assert is_live_fold(date(2026, 6, 1), today=today) is True    # exactly 10
    assert is_live_fold(date(2026, 5, 31), today=today) is False  # 11 days


def test_is_live_fold_accepts_isoformat_string():
    today = date(2026, 6, 7)
    assert is_live_fold("2026-06-06", today=today) is True
    assert is_live_fold("2025-01-01", today=today) is False


# --------------------------------------------------------------------------- #
# seed_serving_recorder — guard prevents backfill pollution                    #
# --------------------------------------------------------------------------- #
class _FakeR:
    """Minimal stand-in for qlib.workflow.R to assert save behaviour offline."""
    def __init__(self):
        self.saved = {}
        self.started = []

    def start(self, **kwargs):
        self.started.append(kwargs)
        fake = self

        class _Ctx:
            def __enter__(self_inner):
                return fake
            def __exit__(self_inner, *a):
                return False
        return _Ctx()

    def save_objects(self, **kwargs):
        self.saved.update(kwargs)


def _df():
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2026-06-02"), "SH600000")],
                                    names=["datetime", "instrument"])
    return pd.DataFrame({"score": [1.0]}, index=idx)


def test_seed_skips_old_fold_without_touching_recorder():
    r = _FakeR()
    today = date(2026, 6, 7)
    seeded = seed_serving_recorder("exp", date(2025, 12, 26), _df(), today=today, _r=r)
    assert seeded is False
    assert r.started == []          # never opened a recorder
    assert r.saved == {}            # never saved pred.pkl


def test_seed_writes_pred_pkl_for_live_fold():
    r = _FakeR()
    today = date(2026, 6, 7)
    seeded = seed_serving_recorder("exp", date(2026, 6, 5), _df(), today=today, _r=r)
    assert seeded is True
    assert len(r.started) == 1
    assert r.started[0]["recorder_name"] == "ensemble_2026-06-05"
    assert "pred.pkl" in r.saved
    assert r.saved["pred.pkl"].equals(_df())


def test_seed_failsoft_on_recorder_error():
    class _BoomR(_FakeR):
        def save_objects(self, **kwargs):
            raise RuntimeError("mlflow down")
    seeded = seed_serving_recorder("exp", date(2026, 6, 5), _df(),
                                   today=date(2026, 6, 7), _r=_BoomR())
    assert seeded is False          # swallows the error, returns False
