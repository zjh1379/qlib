"""Verify candidates() includes per-horizon HorizonPrediction + staleness."""
import numpy as np
import pandas as pd


def _toy_pred_df():
    dates = pd.date_range("2026-05-20", periods=5, freq="B")
    inst = [f"SH{600000+i}" for i in range(30)]
    idx = pd.MultiIndex.from_product([dates, inst], names=["datetime", "instrument"])
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "lgbm_1d": rng.normal(size=len(idx)),
        "alstm_1d": rng.normal(size=len(idx)),
        "tra_1d": rng.normal(size=len(idx)),
        "lgbm_5d": rng.normal(size=len(idx)),
        "alstm_5d": rng.normal(size=len(idx)),
        "tra_5d": rng.normal(size=len(idx)),
        "lgbm_20d": rng.normal(size=len(idx)),
        "alstm_20d": rng.normal(size=len(idx)),
        "tra_20d": rng.normal(size=len(idx)),
        "score": rng.normal(size=len(idx)),
        "consensus": np.abs(rng.normal(size=len(idx))),
    }, index=idx)


def test_candidates_returns_horizons(monkeypatch):
    from app.models import service as svc

    fake = _toy_pred_df()
    monkeypatch.setattr(svc, "load_pred", lambda *a, **kw: fake)
    monkeypatch.setattr(svc, "get_latest_recorder_id", lambda *a, **kw: "fake-rec")
    monkeypatch.setattr(svc, "init_qlib_once", lambda: None)
    monkeypatch.setattr(svc, "get_latest_close_prices", lambda syms: {})
    monkeypatch.setattr("app.core.qlib_adapter.get_filter_metrics", lambda syms: {})
    # No qlib calendar for _get_qlib_latest_date / _next_n_trading_days
    monkeypatch.setattr(svc, "_get_qlib_latest_date", lambda: None)
    svc.invalidate_candidates_cache()

    resp = svc.candidates(top=5, days=3)
    assert "items" in resp
    assert len(resp["items"]) > 0
    for item in resp["items"]:
        assert "horizons" in item
        h = item["horizons"]
        assert set(h.keys()) >= {"1d", "5d", "20d"}
        for hid in ("1d", "5d", "20d"):
            hp = h[hid]
            assert "target_date" in hp
            assert "percentile" in hp
            assert 0 <= hp["percentile"] <= 100


def test_candidates_includes_as_of_and_stale_keys(monkeypatch):
    from app.models import service as svc

    fake = _toy_pred_df()
    monkeypatch.setattr(svc, "load_pred", lambda *a, **kw: fake)
    monkeypatch.setattr(svc, "get_latest_recorder_id", lambda *a, **kw: "fake-rec")
    monkeypatch.setattr(svc, "init_qlib_once", lambda: None)
    monkeypatch.setattr(svc, "get_latest_close_prices", lambda syms: {})
    monkeypatch.setattr("app.core.qlib_adapter.get_filter_metrics", lambda syms: {})
    monkeypatch.setattr(svc, "_get_qlib_latest_date", lambda: None)
    svc.invalidate_candidates_cache()

    resp = svc.candidates(top=5, days=3)
    assert "as_of_date" in resp
    assert "data_latest_date" in resp
    assert "data_stale_days" in resp
    assert resp["data_stale_days"] >= 0


def test_staleness_detected(monkeypatch):
    """If qlib latest > pred latest, data_stale_days > 0."""
    import datetime as dt
    from app.models import service as svc

    fake = _toy_pred_df()
    monkeypatch.setattr(svc, "load_pred", lambda *a, **kw: fake)
    monkeypatch.setattr(svc, "get_latest_recorder_id", lambda *a, **kw: "fake-rec")
    monkeypatch.setattr(svc, "init_qlib_once", lambda: None)
    monkeypatch.setattr(svc, "get_latest_close_prices", lambda syms: {})
    monkeypatch.setattr("app.core.qlib_adapter.get_filter_metrics", lambda syms: {})
    # Simulate qlib having 3 days of data beyond the predictions
    pred_latest = fake.index.get_level_values("datetime").max().date()
    qlib_latest = pred_latest + dt.timedelta(days=7)  # ~5 trading days
    monkeypatch.setattr(svc, "_get_qlib_latest_date", lambda: qlib_latest)
    svc.invalidate_candidates_cache()

    resp = svc.candidates(top=5, days=3)
    # Without qlib calendar available it falls back to calendar days
    assert resp["data_stale_days"] > 0
    assert resp["data_latest_date"] == qlib_latest.isoformat()
