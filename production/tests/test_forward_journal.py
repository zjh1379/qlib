import pandas as pd
from production.research.forward_journal import select_topk


def _pred(date, scores: dict):
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(date), i) for i in scores], names=["datetime", "instrument"])
    return pd.DataFrame({"score": list(scores.values())}, index=idx)


def test_select_topk_picks_highest_ranked_on_date():
    pred = _pred("2026-06-20", {"A": 0.1, "B": 0.9, "C": 0.5, "D": 0.7})
    top = select_topk(pred, pd.Timestamp("2026-06-20"), 3)
    assert list(top["instrument"]) == ["B", "D", "C"]
    assert list(top["rank"]) == [1, 2, 3]
    assert top["score"].iloc[0] == 0.9


def test_select_topk_ignores_other_dates_and_nans():
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-06-19"), "X"), (pd.Timestamp("2026-06-20"), "A"),
         (pd.Timestamp("2026-06-20"), "B"), (pd.Timestamp("2026-06-20"), "C")],
        names=["datetime", "instrument"])
    pred = pd.DataFrame({"score": [9.9, 0.2, float("nan"), 0.8]}, index=idx)
    top = select_topk(pred, pd.Timestamp("2026-06-20"), 2)
    # X is on another date; B is NaN -> dropped. Only A,C remain; C>A.
    assert list(top["instrument"]) == ["C", "A"]
    assert len(top) == 2
