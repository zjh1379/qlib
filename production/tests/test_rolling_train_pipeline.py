from datetime import date

import numpy as np
import pandas as pd
import pytest

from production import rolling_train


def _stub_base(end_date: date, model_id: str) -> list[pd.Series]:
    idx = pd.MultiIndex.from_product(
        [pd.date_range(end_date, periods=1), [f"SH60{i:04d}" for i in range(5)]],
        names=["datetime", "instrument"],
    )
    return [
        pd.Series(np.linspace(0.1, 0.5, 5), index=idx, name=f"{model_id}_1d"),
        pd.Series(np.linspace(-0.2, 0.2, 5), index=idx, name=f"{model_id}_5d"),
        pd.Series(np.linspace(0.0, 0.4, 5), index=idx, name=f"{model_id}_20d"),
    ]


def test_run_once_writes_pred_pkl_via_fallback_when_qlib_unavailable(tmp_path, monkeypatch):
    """When qlib is mocked out, D.features fails and the stacker falls back to
    rank_average. The final pred.pkl still has score + consensus + 9 base columns.

    NOTE: This test only exercises the FALLBACK path; the stacker fit/predict
    path is unit-tested in test_ensemble_stacker.py but not yet integration-tested
    end-to-end. See production/rolling_train.py docstring for the followups list.
    """
    cfg = rolling_train.load_config(rolling_train.REPO_ROOT / "production/configs/rolling_ensemble.yaml")
    monkeypatch.setattr(rolling_train, "init_qlib", lambda c: None)
    monkeypatch.setattr(
        rolling_train,
        "build_universe",
        lambda c, d: ([f"SH60{i:04d}" for i in range(5)], "csi800_pit_test"),
    )

    def _fake_lgbm(cfg, h, universe_name, end):
        return _stub_base(end, "lgbm")[
            [h2.name for h2 in cfg.horizons].index(h.name)
        ]

    monkeypatch.setattr(rolling_train, "train_lgbm_horizon", _fake_lgbm)
    monkeypatch.setattr("production.train_alstm.train_alstm_multihead", lambda c, u, d: _stub_base(d, "alstm"))
    monkeypatch.setattr("production.train_tra.train_tra_multihead", lambda c, u, d: _stub_base(d, "tra"))
    monkeypatch.setattr(rolling_train, "REPO_ROOT", tmp_path)
    (tmp_path / "examples" / "mlruns").mkdir(parents=True, exist_ok=True)

    pred_path = rolling_train.run_once(cfg, date(2026, 5, 10))
    assert pred_path.exists()
    df = pd.read_pickle(pred_path)
    assert "score" in df.columns
    assert "consensus" in df.columns
    base_cols = [c for c in df.columns if c not in {"score", "consensus"}]
    # 9 base columns (3 models x 3 horizons) when stacker fits successfully
    assert len(base_cols) == 9
