# production/tests/test_backfill_pool.py
import pandas as pd
from production.backfill_pool import assemble_score


def _series(d):
    idx = pd.MultiIndex.from_tuples(list(d.keys()), names=["datetime", "instrument"])
    return pd.Series(list(d.values()), index=idx)


def test_assemble_score_ranks_1d_5d_only_and_dedups():
    t = pd.Timestamp("2026-01-02")
    base = pd.DataFrame({
        "lgbm_1d": _series({(t, "A"): 0.9, (t, "B"): 0.1}),
        "lgbm_5d": _series({(t, "A"): 0.8, (t, "B"): 0.2}),
        "lgbm_20d": _series({(t, "A"): -9.0, (t, "B"): 9.0}),  # must be ignored by score
    })
    out = assemble_score(base, ewma_alpha=1.0)  # alpha=1 -> no smoothing
    a = out.xs(t, level="datetime").loc["A", "score"]
    b = out.xs(t, level="datetime").loc["B", "score"]
    assert a > b                      # A better on 1d+5d -> higher score
    assert "score" in out.columns


def test_assemble_score_dedup_keep_last():
    t = pd.Timestamp("2026-01-02")
    base = pd.DataFrame({"lgbm_1d": _series({(t, "A"): 1.0})})
    dup = pd.concat([base, base])     # duplicated index row
    out = assemble_score(dup, ewma_alpha=1.0)
    assert len(out.xs(t, level="datetime")) == 1
