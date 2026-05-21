import pandas as pd
import pytest

from production.ensemble_rank_avg import rank_average


def test_rank_average_two_models():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-05-15"), "SH600000"),
            (pd.Timestamp("2026-05-15"), "SH600001"),
            (pd.Timestamp("2026-05-15"), "SH600002"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {
            "lgbm_5d": [0.30, 0.10, 0.20],  # ranks 1, 3, 2
            "alstm_5d": [0.05, 0.50, 0.10], # ranks 3, 1, 2
        },
        index=idx,
    )
    out = rank_average(df)
    # average rank: 0=(1+3)/2=2; 1=(3+1)/2=2; 2=(2+2)/2=2 -> all tied
    assert out.loc[(pd.Timestamp("2026-05-15"), "SH600000")] == pytest.approx(2.0)
    assert out.loc[(pd.Timestamp("2026-05-15"), "SH600001")] == pytest.approx(2.0)


def test_rank_average_handles_missing_columns():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-05-15"), "SH600000"),
            (pd.Timestamp("2026-05-15"), "SH600001"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {"lgbm_5d": [0.30, 0.10], "alstm_5d": [None, 0.50]},
        index=idx,
    )
    out = rank_average(df)
    # First row only has lgbm -> its score is lgbm's rank (1); second row has both
    # ranks (lgbm:2, alstm:1) -> avg=1.5
    assert out.loc[(pd.Timestamp("2026-05-15"), "SH600000")] == pytest.approx(1.0)
    assert out.loc[(pd.Timestamp("2026-05-15"), "SH600001")] == pytest.approx(1.5)
