import pytest
from app.core.qlib_adapter import init_qlib_once, get_calendar_end
from app.charts.service import get_chart


@pytest.fixture(scope="module", autouse=True)
def _init():
    init_qlib_once()


def test_get_chart_returns_actual_and_predicted():
    end = get_calendar_end()
    payload = get_chart(
        symbol="SH600519",
        start="2025-01-01",
        end=str(end),
        with_pred=True,
        experiment="daily_cn_fresh",
    )
    assert payload.symbol == "SH600519"
    assert len(payload.actual) > 100
    # predicted is shifted by 2 trading days, so should be (actual_len - 2)
    assert abs(len(payload.predicted) - (len(payload.actual) - 2)) <= 5
    # forecast contains future bars (1 or 2 of them)
    assert 0 <= len(payload.forecast) <= 2


def test_get_chart_without_pred_returns_only_actual():
    end = get_calendar_end()
    payload = get_chart(
        symbol="SH600519",
        start="2025-04-01",
        end=str(end),
        with_pred=False,
        experiment="daily_cn_fresh",
    )
    assert len(payload.actual) > 5
    assert len(payload.predicted) == 0
    assert len(payload.forecast) == 0


def test_get_chart_unknown_symbol_raises():
    from app.core.exceptions import NotFoundError

    with pytest.raises(NotFoundError):
        get_chart(symbol="SH999999", start="2025-01-01", end="2025-02-01", with_pred=False)


def test_predicted_bar_open_equals_prior_actual_close():
    end = get_calendar_end()
    payload = get_chart(
        symbol="SH600519",
        start="2025-04-01",
        end=str(end),
        with_pred=True,
        experiment="daily_cn_fresh",
    )
    actual_by_time = {b.time: b for b in payload.actual}
    actual_times = sorted(actual_by_time.keys())
    for pred_bar in payload.predicted[:5]:
        # find the actual bar 1 day before pred_bar.time in trading-day terms
        idx = actual_times.index(pred_bar.time)
        prior_actual = actual_by_time[actual_times[idx - 1]]
        assert abs(pred_bar.open - prior_actual.close) < 1e-6, (
            f"pred[{pred_bar.time}].open should equal actual[{actual_times[idx-1]}].close"
        )


def test_forecast_bars_chain_correctly():
    """forecast[0].open == actual[-1].close, forecast[1].open == forecast[0].close (chained)."""
    end = get_calendar_end()
    payload = get_chart(
        symbol="SH600519",
        start="2025-04-01",
        end=str(end),
        with_pred=True,
        experiment="daily_cn_fresh",
    )
    if len(payload.forecast) < 1:
        pytest.skip("no forecast bars produced — score may be missing for last signal date")
    last_actual_close = payload.actual[-1].close
    assert abs(payload.forecast[0].open - last_actual_close) < 1e-6, \
        f"forecast[0].open should equal last actual close"
    if len(payload.forecast) >= 2:
        assert abs(payload.forecast[1].open - payload.forecast[0].close) < 1e-6, \
            f"forecast[1].open should equal forecast[0].close (chained prediction)"
