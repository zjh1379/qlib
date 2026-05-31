import json
import pandas as pd
import pytest
from production.backtest.run import build_report, extract_score_series


def _series(d):
    idx = pd.MultiIndex.from_tuples(list(d.keys()), names=["datetime", "instrument"])
    return pd.Series(list(d.values()), index=idx)


def test_extract_score_series_from_dataframe():
    df = pd.DataFrame({"score": [1.0, 2.0], "other": [9, 9]},
                      index=pd.MultiIndex.from_tuples(
                          [(pd.Timestamp("2024-01-02"), "A"), (pd.Timestamp("2024-01-02"), "B")],
                          names=["datetime", "instrument"]))
    s = extract_score_series(df, "score")
    assert list(s.values) == [1.0, 2.0]


def test_build_report_has_metrics_and_params():
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    scores = _series({(dates[0], "A"): 1.0, (dates[1], "A"): 1.0})
    fwd = _series({(dates[0], "A"): 0.01, (dates[1], "A"): 0.02})
    rep = build_report(scores, fwd, policy_name="daily", top_k=1, period=5,
                       exit_k=2, capital=100_000.0, profile="small")
    assert "metrics" in rep and "params" in rep
    assert rep["params"]["policy"] == "daily"
    assert rep["metrics"]["n_days"] == 2
    # JSON-serializable
    json.dumps(rep)


def test_load_sector_map(tmp_path):
    p = tmp_path / "ind.parquet"
    pd.DataFrame({"instrument": ["SH600000", "SZ000001"],
                  "industry": ["银行", "银行"]}).to_parquet(p, index=False)
    from production.backtest.run import load_sector_map
    m = load_sector_map(str(p))
    assert m["SH600000"] == "银行"
    assert m["SZ000001"] == "银行"
