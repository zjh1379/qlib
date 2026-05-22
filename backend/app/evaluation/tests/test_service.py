import pytest

from app.evaluation.service import list_recorders_with_summary
from app.core.qlib_adapter import init_qlib_once


@pytest.fixture(scope="module")
def qlib_ready():
    try:
        init_qlib_once()
    except Exception as exc:
        pytest.skip(f"qlib not initializable: {exc}")


def test_list_recorders_returns_at_least_one_known(qlib_ready):
    summaries = list_recorders_with_summary()
    # The dev environment must have at least the daily_cn_fresh recorder
    assert any(s.experiment == "daily_cn_fresh" for s in summaries), \
        "expected daily_cn_fresh experiment to have at least one recorder"


def test_summary_fields_are_populated(qlib_ready):
    summaries = list_recorders_with_summary()
    if not summaries:
        pytest.skip("no recorders available")
    s = summaries[0]
    assert s.recorder_id
    assert s.experiment
    assert s.run_name
    assert s.created_at
    # pred_start/end/rows may be None on errors but should usually be set
    if s.pred_rows is not None:
        assert s.pred_rows > 0
    assert s.has_eval is False  # cache empty on first call


from app.evaluation.service import evaluate_recorder, list_recorders_with_summary


def test_evaluate_daily_cn_fresh_recorder(qlib_ready):
    summaries = list_recorders_with_summary()
    target = next((s for s in summaries if s.experiment == "daily_cn_fresh"), None)
    if target is None:
        pytest.skip("no daily_cn_fresh recorder")

    result = evaluate_recorder(target.recorder_id, top_k=30, cost_bps=10)
    assert result.recorder_id == target.recorder_id
    assert result.experiment == "daily_cn_fresh"
    assert result.sample_size > 100, "expected >100 (date,symbol) pairs after label join"
    # IC for a real model should be in [-0.1, 0.1] — sanity bound
    assert -0.1 < result.scorecard.ic_mean < 0.1
    # IR should be finite (could be negative if the model is bad)
    assert -10 < result.scorecard.ir < 10
    # Acceptance has 5 detail keys per spec
    assert set(result.acceptance.details.keys()) == {
        "ic_mean", "ir", "max_drawdown", "daily_turnover", "regimes_all_positive",
    }


def test_evaluate_recorder_is_cached(qlib_ready):
    summaries = list_recorders_with_summary()
    target = next((s for s in summaries if s.experiment == "daily_cn_fresh"), None)
    if target is None:
        pytest.skip("no daily_cn_fresh recorder")

    import time
    # Force a cold start so the timing comparison is meaningful regardless of test order.
    t0 = time.time()
    evaluate_recorder(target.recorder_id, force_refresh=True)
    t_first = time.time() - t0
    t0 = time.time()
    evaluate_recorder(target.recorder_id)
    t_cached = time.time() - t0
    # Cache hit should be at least 50x faster than the first call.
    assert t_cached * 50 < t_first, f"first={t_first:.2f}s cached={t_cached:.3f}s — caching not effective"


def test_force_refresh_bypasses_cache(qlib_ready):
    summaries = list_recorders_with_summary()
    target = next((s for s in summaries if s.experiment == "daily_cn_fresh"), None)
    if target is None:
        pytest.skip("no daily_cn_fresh recorder")
    r1 = evaluate_recorder(target.recorder_id)
    r2 = evaluate_recorder(target.recorder_id, force_refresh=True)
    # Same recorder, same data, same metrics — but computed_at differs.
    assert r1.scorecard.ic_mean == r2.scorecard.ic_mean
    assert r1.computed_at != r2.computed_at
