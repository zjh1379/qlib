import pandas as pd, pytest
from production.qlib_features import _normalize


def _raw(idx_tuples, idx_names, cols, vals):
    idx = pd.MultiIndex.from_tuples(idx_tuples, names=idx_names)
    return pd.DataFrame(vals, index=idx, columns=cols)


def test_normalize_swaps_instrument_first_and_renames():
    raw = _raw([("SH600000", pd.Timestamp("2024-01-02")),
                ("SH600000", pd.Timestamp("2024-01-03"))],
               ["instrument", "datetime"], ["Ref($open,-1)"], [[1.0], [2.0]])
    out = _normalize(raw, ["entry_open"])
    assert list(out.columns) == ["entry_open"]
    assert out.index.names == ["datetime", "instrument"]
    assert out.loc[(pd.Timestamp("2024-01-02"), "SH600000"), "entry_open"] == 1.0


def test_normalize_multi_column_by_position():
    raw = _raw([("SH600000", pd.Timestamp("2024-01-02"))],
               ["instrument", "datetime"], ["$open", "$close"], [[10.0, 11.0]])
    out = _normalize(raw, ["open", "close"])
    assert list(out.columns) == ["open", "close"]
    assert out.iloc[0]["close"] == 11.0


def test_normalize_already_datetime_first_no_swap():
    raw = _raw([(pd.Timestamp("2024-01-02"), "SH600000")],
               ["datetime", "instrument"], ["x"], [[5.0]])
    out = _normalize(raw, ["v"])
    assert out.index.names == ["datetime", "instrument"]
    assert out.iloc[0]["v"] == 5.0


def test_normalize_column_count_mismatch_raises():
    raw = _raw([("SH600000", pd.Timestamp("2024-01-02"))],
               ["instrument", "datetime"], ["a", "b"], [[1.0, 2.0]])
    with pytest.raises(ValueError):
        _normalize(raw, ["only_one"])
