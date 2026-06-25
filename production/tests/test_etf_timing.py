import pandas as pd, pytest
from production.backtest.etf_timing import simulate_etf_timing
from production.backtest.costs import cost_model


def test_full_exposure_tracks_etf_minus_flip_cost():
    dates = pd.bdate_range("2024-01-02", periods=4)
    etf = pd.Series([0.01, 0.01, 0.01, 0.01], index=dates)
    exp = pd.Series([1.0, 1.0, 1.0, 1.0], index=dates)
    led = simulate_etf_timing(etf, exp, cost_model("etf"), capital=10_000.0)
    assert led["turnover"].iloc[0] == pytest.approx(0.0)  # pos starts 0 (exposure shifted)
    assert led["turnover"].iloc[1] == pytest.approx(1.0)  # flip 0->1 (one-time cost)
    assert led["net"].iloc[2] == pytest.approx(0.01)      # invested, no flip, no cost


def test_zero_exposure_is_flat():
    dates = pd.bdate_range("2024-01-02", periods=3)
    etf = pd.Series([0.05, -0.05, 0.05], index=dates)
    exp = pd.Series([0.0, 0.0, 0.0], index=dates)
    led = simulate_etf_timing(etf, exp, cost_model("etf"))
    assert led["net"].abs().sum() == pytest.approx(0.0)
