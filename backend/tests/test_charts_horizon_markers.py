"""Smoke tests for HorizonMarker schema."""
from app.charts.schemas import ChartPayload, HorizonMarker


def test_horizon_marker_default_values():
    hm = HorizonMarker(
        horizon="5d", target_date="2026-06-03", target_price=125.4,
        pred_return=0.032, percentile=98.6,
    )
    assert hm.model_agreement is None
    assert hm.raw_scores == {}


def test_chart_payload_default_empty_horizon_markers():
    p = ChartPayload(symbol="SH600519", actual=[], predicted=[], forecast=[])
    assert p.horizon_markers == []


def test_chart_payload_with_horizon_markers():
    p = ChartPayload(
        symbol="SH600519", actual=[], predicted=[], forecast=[],
        horizon_markers=[
            HorizonMarker(
                horizon="5d", target_date="2026-06-03", target_price=125.4,
                pred_return=0.032, percentile=98.6, model_agreement=0.67,
                raw_scores={"lgbm": 0.04, "alstm": 0.02},
            ),
        ],
    )
    assert len(p.horizon_markers) == 1
    assert p.horizon_markers[0].horizon == "5d"
