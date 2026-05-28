"""Smoke test for backfill_calibration script using fixture data.

The dataframe-level logic is tested in test_calibration.py. This test
verifies the orchestration function _fit_and_save runs end-to-end on a
fake pred/label pair.
"""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd

from production.backfill_calibration import _fit_and_save


def _fixture_pred_label(seed=0):
    rng = np.random.default_rng(seed)
    n_dates, n_inst = 60, 30
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="B")
    instruments = [f"S{i:03d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dates, instruments],
                                      names=["datetime", "instrument"])
    pred = pd.DataFrame({
        "lgbm_1d": rng.normal(size=len(idx)),
        "alstm_1d": rng.normal(size=len(idx)),
        "tra_1d": rng.normal(size=len(idx)),
        "lgbm_5d": rng.normal(size=len(idx)),
        "alstm_5d": rng.normal(size=len(idx)),
        "tra_5d": rng.normal(size=len(idx)),
    }, index=idx)
    label = pd.DataFrame({
        "label_1d": rng.normal(size=len(idx)),
        "label_5d": rng.normal(size=len(idx)),
    }, index=idx)
    return pred, label


def test_fit_and_save_writes_pickle(tmp_path):
    pred, label = _fixture_pred_label()
    out_path = tmp_path / "latest_calibration.pkl"
    _fit_and_save(pred, label, out_path, trained_at="2026-05-22")
    assert out_path.exists()
    with out_path.open("rb") as f:
        payload = pickle.load(f)
    assert set(payload["maps"].keys()) == {"1d", "5d"}
    assert payload["meta"]["trained_at"] == "2026-05-22"
    assert "saved_at" in payload["meta"]
    assert payload["meta"]["n_rows"] > 0
