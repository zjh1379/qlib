import pytest
import numpy as np
import pandas as pd
from production.backtest.metrics_net import net_metrics, net_regime


def _mk_daily(n=252, mean=0.001, std=0.01, turnover=0.1, cost=0.0001, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    net = rng.normal(mean, std, n)
    gross = net + cost
    return pd.DataFrame(
        {"gross": gross, "cost": cost, "net": net,
         "turnover": turnover, "nav": (1 + pd.Series(net, index=dates)).cumprod() * 1e5},
        index=pd.Index(dates, name="datetime"),
    )


def test_net_metrics_keys_and_types():
    out = net_metrics(_mk_daily())
    for k in ["net_cagr", "gross_cagr", "net_ir", "max_drawdown",
              "avg_turnover", "cost_drag_annual", "win_rate", "n_days"]:
        assert k in out
    assert out["n_days"] == 252
    assert out["avg_turnover"] == pytest.approx(0.1, rel=1e-9)


def test_positive_drift_has_positive_cagr():
    out = net_metrics(_mk_daily(mean=0.002, std=0.005, seed=1))
    assert out["net_cagr"] > 0
    assert out["net_ir"] > 0


def test_empty_daily_returns_nan():
    out = net_metrics(pd.DataFrame(columns=["gross", "cost", "net", "turnover", "nav"]))
    assert out["n_days"] == 0
    assert np.isnan(out["net_ir"])


def test_net_regime_splits():
    daily = _mk_daily(n=200)
    segs = net_regime(daily, [("2024-01-01", "2024-03-31"), ("2024-04-01", "2024-12-31")])
    assert len(segs) == 2
    for _, m in segs.items():
        assert "net_ir" in m


def test_tail_stats_basic():
    from production.backtest.metrics_net import tail_stats
    s = pd.Series([-0.10, -0.02, 0.01, 0.03, 0.20])
    out = tail_stats(s)
    assert out["worst"] == pytest.approx(-0.10)
    assert out["neg_period_pct"] == pytest.approx(0.4)
    assert out["n"] == 5
    assert out["ret_p10"] == pytest.approx(float(s.quantile(0.10)))


def test_tail_stats_empty_is_nan():
    from production.backtest.metrics_net import tail_stats
    out = tail_stats(pd.Series([], dtype=float))
    assert out["n"] == 0
    assert out["worst"] != out["worst"]  # nan


def test_period_metrics_annualizes_by_bars_per_period():
    import pandas as pd
    from production.backtest.metrics_net import period_metrics
    r = pd.Series([0.10, 0.10])           # 2 periods, 5 trading-days each -> 10 days
    m = period_metrics(r, bars_per_period=5)
    assert m["n_periods"] == 2
    assert m["max_dd"] == pytest.approx(0.0)
    assert m["net_cagr"] == pytest.approx(1.21 ** (252 / 10) - 1)
    assert m["win"] == pytest.approx(1.0)


def test_period_metrics_drawdown_and_calmar():
    import pandas as pd
    from production.backtest.metrics_net import period_metrics
    r = pd.Series([0.2, -0.5, 0.1])       # eq 1.2, 0.6, 0.66 ; peak 1.2 -> dd -0.5
    m = period_metrics(r, bars_per_period=1)
    assert m["max_dd"] == pytest.approx(-0.5)
    assert m["calmar"] == pytest.approx(m["net_cagr"] / 0.5)


def test_period_metrics_empty_is_nan():
    import pandas as pd
    from production.backtest.metrics_net import period_metrics
    m = period_metrics(pd.Series([], dtype=float))
    assert m["n_periods"] == 0 and m["net_cagr"] != m["net_cagr"]
