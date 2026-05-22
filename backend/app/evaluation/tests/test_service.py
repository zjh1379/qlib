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


def test_regime_labels_align_when_some_segments_empty(qlib_ready):
    """Regression: if regime_split drops an empty segment from its return dict,
    the loop in evaluate_recorder must still correctly label remaining segments
    (no off-by-one from positional zip)."""
    summaries = list_recorders_with_summary()
    target = next((s for s in summaries if s.experiment == "daily_cn_fresh"), None)
    if target is None:
        pytest.skip("no daily_cn_fresh recorder")

    result = evaluate_recorder(target.recorder_id, force_refresh=True)
    # daily_cn_fresh predictions start in 2025 — none of the 5 spec regimes
    # (2018-2024) overlap; only the synthesized "Recent (full window)" segment
    # should appear.
    labels = [r.label for r in result.regimes]
    assert "Recent (full window)" in labels, f"expected 'Recent' segment, got {labels}"
    # Every regime label must match an entry from _REGIME_SEGMENTS or be the Recent catch-all
    from app.evaluation.service import _REGIME_SEGMENTS
    valid = {l for l, _, _ in _REGIME_SEGMENTS} | {"Recent (full window)"}
    for r in result.regimes:
        assert r.label in valid, f"regime label {r.label!r} not in spec set"
        # Each segment's date range must match a known regime spec (or the synthetic recent)
        if r.label != "Recent (full window)":
            spec = next(seg for seg in _REGIME_SEGMENTS if seg[0] == r.label)
            assert (r.start, r.end) == (spec[1], spec[2]), \
                f"regime {r.label}: dates {r.start}..{r.end} don't match spec {spec[1]}..{spec[2]}"


from app.evaluation.service import compare_recorders


def test_compare_same_recorder_against_itself(qlib_ready):
    summaries = list_recorders_with_summary()
    target = next((s for s in summaries if s.experiment == "daily_cn_fresh"), None)
    if target is None:
        pytest.skip("no daily_cn_fresh recorder")

    cmp = compare_recorders(target.recorder_id, target.recorder_id)
    # IC delta and IR delta must be zero (same recorder)
    assert abs(cmp.ic_delta) < 1e-9
    assert abs(cmp.ir_delta) < 1e-9
    # Paired t-test on identical series: t = nan or 0; p ≈ 1
    # Just assert the verdict says no significant difference
    assert cmp.significant_at_05 is False
    assert "no significant difference" in cmp.verdict.lower()


def test_compare_different_recorders_returns_full_eval(qlib_ready):
    summaries = list_recorders_with_summary()
    # We need two distinct recorders; if there's only one, skip.
    if len(summaries) < 2:
        pytest.skip("need >=2 recorders to compare")
    a = summaries[0].recorder_id
    b = summaries[1].recorder_id
    if a == b:
        pytest.skip("first two recorders are identical")

    cmp = compare_recorders(a, b)
    assert cmp.a.recorder_id == a
    assert cmp.b.recorder_id == b
    # IC delta = b.ic_mean - a.ic_mean (per schema)
    assert cmp.ic_delta == pytest.approx(cmp.b.scorecard.ic_mean - cmp.a.scorecard.ic_mean, abs=1e-9)
    assert cmp.ir_delta == pytest.approx(cmp.b.scorecard.ir - cmp.a.scorecard.ir, abs=1e-9)
    # verdict is one of 3 expected strings
    assert cmp.verdict in (
        "b significantly better",
        "a significantly better",
        "no significant difference",
    )
