import pandas as pd
from app.models.service import _build_screen_items, _window_dates


def _mk_df():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2026-05-10", "2026-05-14"), ["SH600000", "SH600001"]],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {"score": [0.10, -0.05, 0.11, -0.04, 0.13, -0.03, 0.12, -0.02, 0.14, -0.01]},
        index=idx,
    )
    return df


def test_window_dates_ascending_last_k():
    df = _mk_df()
    wd = _window_dates(df, k=3)
    assert wd == ["2026-05-12", "2026-05-13", "2026-05-14"]
    assert _window_dates(df, k=99)[0] == "2026-05-10"


def test_daily_arrays_aligned_and_typed():
    df = _mk_df()
    items = _build_screen_items(df, top=2, days=5, min_top=0, name_map={})
    top = next(it for it in items if it.symbol == "SH600000")
    assert top.daily_ranks == [1, 1, 1, 1, 1]
    assert len(top.daily_scores) == 5
    assert top.daily_scores[-1] == 0.14
    other = next(it for it in items if it.symbol == "SH600001")
    assert other.daily_ranks == [2, 2, 2, 2, 2]


def test_daily_arrays_fill_none_for_missing_day():
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-05-10"), "SH600000"),
         (pd.Timestamp("2026-05-12"), "SH600000"),
         (pd.Timestamp("2026-05-10"), "SH600001"),
         (pd.Timestamp("2026-05-12"), "SH600001")],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame({"score": [0.2, 0.3, 0.1, 0.1]}, index=idx)
    items = _build_screen_items(df, top=2, days=2, min_top=0, name_map={})
    for it in items:
        assert len(it.daily_ranks) == 2
        assert all(r is None or isinstance(r, int) for r in it.daily_ranks)
