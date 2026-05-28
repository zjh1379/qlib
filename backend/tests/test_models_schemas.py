"""Schema tests for HorizonPrediction + ScreenItem.horizons."""
from app.models.schemas import (
    CandidatesResponse,
    HorizonPrediction,
    ScreenItem,
)


def test_horizon_prediction_defaults():
    h = HorizonPrediction(target_date="2026-06-03", percentile=95.0)
    assert h.pred_return is None
    assert h.model_agreement is None
    assert h.raw_scores == {}


def test_screen_item_horizons_default_empty():
    it = ScreenItem(
        rank=1, symbol="SH600519", name="贵州茅台",
        score_today=0.1, score_avg=0.1, rank_avg=1.0, days_in_top=5,
    )
    assert it.horizons == {}


def test_screen_item_with_horizons():
    it = ScreenItem(
        rank=1, symbol="SH600519", name="贵州茅台",
        score_today=0.1, score_avg=0.1, rank_avg=1.0, days_in_top=5,
        horizons={
            "5d": HorizonPrediction(
                target_date="2026-06-03", pred_return=0.032, percentile=98.6,
            )
        },
    )
    assert it.horizons["5d"].pred_return == 0.032


def test_candidates_response_staleness_defaults():
    resp = CandidatesResponse(
        experiment="exp", recorder_id="abc", latest_date="2026-05-22",
        window_days=5, universe_size=800, items=[],
    )
    assert resp.as_of_date is None
    assert resp.data_latest_date is None
    assert resp.data_stale_days == 0
