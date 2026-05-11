from app.charts.schemas import CandleBar, PredictionBar, ChartPayload


def test_candle_bar_valid():
    bar = CandleBar(time="2026-05-08", open=10.0, high=11.0, low=9.5, close=10.5, volume=1000.0)
    assert bar.time == "2026-05-08"
    assert bar.close == 10.5


def test_prediction_bar_valid():
    pb = PredictionBar(time="2026-05-11", open=10.5, high=10.7, low=10.4, close=10.6, score=0.02)
    assert pb.score == 0.02


def test_chart_payload_assembles():
    payload = ChartPayload(
        symbol="SH600519",
        actual=[CandleBar(time="2026-05-08", open=1.0, high=2.0, low=0.5, close=1.5, volume=100.0)],
        predicted=[],
        forecast=[],
        meta={"last_actual_date": "2026-05-08", "experiment": "daily_cn_fresh"},
    )
    assert payload.symbol == "SH600519"
    assert len(payload.actual) == 1
