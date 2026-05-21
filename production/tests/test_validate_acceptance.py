import json
from pathlib import Path

from production.validate_acceptance import check_acceptance


def test_check_acceptance_returns_status_dict(tmp_path: Path):
    scorecard = {
        "ic_mean": 0.032,
        "ric_mean": 0.026,
        "icir": 0.45,
        "top_bottom_spread_monthly": 1.8,
        "annual_excess_return": 0.18,
        "ir": 2.6,
        "max_drawdown": -0.12,
        "daily_turnover": 0.18,
    }
    out = check_acceptance(scorecard, regime_irs={"a": 0.5, "b": 0.3, "c": 0.7, "d": 0.4, "e": 0.6})
    assert out["passed"] is True
    assert all(out["details"].values())


def test_check_acceptance_flags_low_ic():
    scorecard = {
        "ic_mean": 0.025,
        "ric_mean": 0.026,
        "icir": 0.45,
        "top_bottom_spread_monthly": 1.8,
        "annual_excess_return": 0.18,
        "ir": 2.6,
        "max_drawdown": -0.12,
        "daily_turnover": 0.18,
    }
    out = check_acceptance(scorecard, regime_irs={"a": 0.5})
    assert out["passed"] is False
    assert out["details"]["ic_mean"] is False


def test_check_acceptance_flags_negative_regime():
    scorecard = {
        "ic_mean": 0.032, "ric_mean": 0.026, "icir": 0.45,
        "top_bottom_spread_monthly": 1.8, "annual_excess_return": 0.18,
        "ir": 2.6, "max_drawdown": -0.12, "daily_turnover": 0.18,
    }
    out = check_acceptance(scorecard, regime_irs={"a": -0.1, "b": 0.3})
    assert out["passed"] is False
    assert out["details"]["regimes_all_positive"] is False
