import numpy as np
import pandas as pd
import pytest
from production.score_utils import score_of, rebuild_2model, calmar


def _mk(prefix, dates, insts, seed):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([dates, insts], names=["datetime", "instrument"])
    return pd.DataFrame({f"{prefix}_{h}": rng.standard_normal(len(idx))
                         for h in ("1d", "5d", "20d")}, index=idx)


def test_score_of_normalizes_series_and_frame():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A", "B"]])
    s = pd.Series([1.0, 2.0, 3.0, np.nan], index=idx)
    out = score_of(s)
    assert out.name == "score"
    assert list(out.index.names) == ["datetime", "instrument"]
    assert len(out) == 3                       # NaN dropped
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    assert score_of(df).name == "score" and len(score_of(df)) == 4


def test_rebuild_2model_blends_lgbm_and_alstm():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    insts = ["A", "B", "C", "D"]
    out = rebuild_2model(_mk("lgbm", dates, insts, 1), _mk("alstm", dates, insts, 2))
    assert isinstance(out, pd.Series) and out.name == "score"
    assert list(out.index.names) == ["datetime", "instrument"]
    assert out.notna().all()
    assert len(out) == len(dates) * len(insts)


def test_calmar_nan_safe():
    assert calmar({"net_cagr": 0.30, "max_drawdown": -0.40}) == pytest.approx(0.75)
    assert np.isnan(calmar({"net_cagr": 0.30, "max_drawdown": 0.0}))
    assert np.isnan(calmar({"net_cagr": 0.30, "max_drawdown": None}))
