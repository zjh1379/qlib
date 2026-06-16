from app.models.schemas import (
    ScreenItem, CandidatesResponse, RecomputeProgress, RecomputeJob,
    RecomputeRequest, RecomputeTriggerResponse,
)


def test_screen_item_accepts_daily_arrays():
    it = ScreenItem(
        rank=1, symbol="SH600000", score_today=0.1, score_avg=0.1,
        rank_avg=1.0, days_in_top=5,
        daily_ranks=[3, None, 1], daily_scores=[0.1, None, 0.2],
    )
    assert it.daily_ranks == [3, None, 1]
    assert it.daily_scores == [0.1, None, 0.2]


def test_candidates_response_has_window_dates():
    r = CandidatesResponse(
        experiment="e", recorder_id="r", latest_date="2026-06-16",
        window_days=20, universe_size=800, items=[],
        window_dates=["2026-06-12", "2026-06-13"],
    )
    assert r.window_dates == ["2026-06-12", "2026-06-13"]


def test_recompute_models_roundtrip():
    p = RecomputeProgress(phase="metrics", percent=60, message="x")
    job = RecomputeJob(job_id="j", status="running", started_at="t",
                       view="ensemble", models=["lgbm_5d"], progress=p)
    assert job.progress.percent == 60
    assert RecomputeRequest(view="ensemble", models=[]).models == []
    assert RecomputeTriggerResponse(status="started", job_id="j").job_id == "j"
