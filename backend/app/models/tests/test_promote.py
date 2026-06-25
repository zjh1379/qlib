import pandas as pd
import pytest
from app.models import service as svc


def test_promote_copies_pred_into_production(monkeypatch):
    saved = {}
    fake_pred = pd.DataFrame({"score": [1.0, 2.0]})

    class _Rec:
        def load_object(self, key): return fake_pred
    class _Started:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    class _R:
        @staticmethod
        def get_recorder(recorder_id=None, experiment_name=None): return _Rec()
        @staticmethod
        def start(experiment_name=None, recorder_name=None):
            saved["exp"] = experiment_name; saved["name"] = recorder_name; return _Started()
        @staticmethod
        def save_objects(**kw): saved["pred_rows"] = len(kw["pred.pkl"])
    monkeypatch.setattr(svc, "R", _R, raising=False)
    monkeypatch.setattr(svc, "init_qlib_once", lambda *a, **k: None, raising=False)

    out = svc.promote_candidate("cand_rec_1", candidate_experiment="exp_candidates", production_experiment="exp")
    assert out["status"] == "promoted"
    assert saved["exp"] == "exp"
    assert saved["name"].startswith("promoted_")
    assert saved["pred_rows"] == 2
