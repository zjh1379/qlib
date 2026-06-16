import pandas as pd
from production.reblend import series_from_recorders


class _FakeRec:
    def __init__(self, name, objs):
        self.id = name
        self.info = {"name": name}
        self._objs = objs
    def load_object(self, key):
        return self._objs[key]


def _series(vals):
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2026-01-05"]), ["SH600000", "SZ000001"]],
                                     names=["datetime", "instrument"])
    return pd.Series(vals, index=idx)


def test_series_from_recorders_matches_model_horizon_names():
    recs = [
        _FakeRec("alstm_5d_2026-06-12", {"pred_5d.pkl": _series([0.1, 0.2])}),
        _FakeRec("tra_5d_2026-06-12", {"pred_5d.pkl": pd.DataFrame({"score": _series([0.3, 0.4])})}),
        _FakeRec("lgbm_5d_2026-06-12", {"pred_5d.pkl": _series([0.9, 0.9])}),  # excluded by model_ids
        _FakeRec("ensemble_2026-06-12", {}),  # ignored
    ]
    out = series_from_recorders(recs, end_str="2026-06-12", model_ids=("alstm", "tra"), horizons=("5d",))
    cols = sorted(s.name for s in out)
    assert cols == ["alstm_5d", "tra_5d"]
    tra = next(s for s in out if s.name == "tra_5d")
    assert list(tra.values) == [0.3, 0.4]


def test_series_from_recorders_empty_when_no_match():
    out = series_from_recorders([], end_str="2026-06-12", model_ids=("alstm",), horizons=("5d",))
    assert out == []
