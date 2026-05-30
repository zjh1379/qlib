import numpy as np
import pandas as pd
from production.backtest.sweep import run_sweep


def _mk():
    dates = pd.bdate_range("2024-01-02", periods=40)
    stocks = [f"S{i}" for i in range(25)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    g = np.random.default_rng(0)
    base = {s: g.normal(0, 1) for s in stocks}
    scores = pd.Series([base[s] + g.normal(0, 0.3) for (_, s) in idx], index=idx)
    fwd = pd.Series(g.normal(0.0005, 0.01, len(idx)), index=idx)
    return scores, fwd


def test_run_sweep_grid_shape_and_columns():
    scores, fwd = _mk()
    grid = run_sweep(
        scores, fwd,
        policies=["daily", "banded"],
        top_ks=[5, 10],
        periods=[5],
        capitals=[50_000, 100_000],
        profile="small",
    )
    # 2 policies x 2 top_k x 2 capital (period only matters for 'fixed') = 8 rows
    assert len(grid) == 8
    for col in ["policy", "top_k", "capital", "net_ir", "avg_turnover", "net_cagr"]:
        assert col in grid.columns


def test_run_sweep_banded_turnover_below_daily_on_average():
    scores, fwd = _mk()
    grid = run_sweep(scores, fwd, policies=["daily", "banded"], top_ks=[10],
                     periods=[5], capitals=[100_000], profile="small")
    t_daily = grid[grid.policy == "daily"]["avg_turnover"].mean()
    t_banded = grid[grid.policy == "banded"]["avg_turnover"].mean()
    assert t_banded < t_daily
