import pandas as pd
from production.rolling_train import _stack_score


def test_stack_score_produces_score_column():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05", "2026-01-06"]), ["SH600000", "SZ000001", "SH601318"]],
        names=["datetime", "instrument"],
    )
    base = pd.DataFrame({"lgbm_5d": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}, index=idx)
    labels = pd.Series([0.01, -0.01, 0.02, 0.0, 0.03, -0.02], index=idx, name="y")
    out = _stack_score(base, labels)
    assert "score" in out.columns
    assert len(out) == len(base)
    assert out["score"].notna().any()
