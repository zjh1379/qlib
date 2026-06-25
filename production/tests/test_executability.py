import pandas as pd, pytest
from production.backtest.executability import (
    buyable_mask, gate_scores, selection_bias_split)


def _ohlc(rows):
    # rows: (date_str, instrument, entry_open, prev_close)
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), i) for d, i, *_ in rows], names=["datetime", "instrument"])
    return pd.DataFrame({"entry_open": [r[2] for r in rows],
                         "prev_close": [r[3] for r in rows]}, index=idx)


def test_buyable_mask_main_board_10pct():
    # SH600000 main 10%: cap=11.0. open 10.5<11 -> buyable; open 11.0==cap -> not
    o = _ohlc([("2024-01-02", "SH600000", 10.5, 10.0),
               ("2024-01-02", "SZ000001", 11.0, 10.0)])
    m = buyable_mask(o)
    assert bool(m.xs("2024-01-02", level="datetime")["SH600000"]) is True
    assert bool(m.xs("2024-01-02", level="datetime")["SZ000001"]) is False


def test_buyable_mask_chinext_20pct():
    # SZ300750 ChiNext 20%: cap=12.0; open 11.5 buyable (would fail on a main-board cap)
    o = _ohlc([("2024-01-02", "SZ300750", 11.5, 10.0)])
    assert bool(buyable_mask(o).iloc[0]) is True


def test_gate_scores_drops_unbuyable_promoting_next_rank():
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), i) for i in ["A", "B", "C"]],
        names=["datetime", "instrument"])
    scores = pd.Series([0.9, 0.8, 0.7], index=idx)
    buyable = pd.Series([False, True, True], index=idx)  # A unbuyable
    g = gate_scores(scores, buyable)
    assert "A" not in g.index.get_level_values("instrument")
    assert g.xs("2024-01-02", level="datetime").idxmax() == "B"  # next rank promoted


def test_gate_scores_keeps_unknown():
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2024-01-02"), "A")],
                                    names=["datetime", "instrument"])
    scores = pd.Series([0.9], index=idx)
    g = gate_scores(scores, pd.Series(dtype=bool))  # no info -> keep
    assert len(g) == 1


def test_selection_bias_split_detects_missed_winners():
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), i) for i in ["A", "B", "C"]],
        names=["datetime", "instrument"])
    scores = pd.Series([0.9, 0.8, 0.1], index=idx)
    fwd = pd.Series([0.10, 0.01, 0.0], index=idx)   # A +10% (unbuyable), B +1% (buyable)
    buyable = pd.Series([False, True, True], index=idx)
    out = selection_bias_split(scores, fwd, buyable, top_k=2, period=1)
    assert out["n_picks"] == 2 and out["n_unbuyable"] == 1
    assert out["unbuyable_mean_ret"] == pytest.approx(0.10)
    assert out["buyable_mean_ret"] == pytest.approx(0.01)
    assert out["edge_missed"] == pytest.approx(0.09)  # winners we miss
