"""Single-algorithm re-blend: retrain one base model and re-blend it with the
OTHER models' cached predictions (loaded from their existing recorders) into a
CANDIDATE ensemble. See docs/superpowers/plans/2026-06-16-training-studio-p3a-single-algo.md."""
from __future__ import annotations

import logging

import pandas as pd

_log = logging.getLogger("reblend")


def series_from_recorders(recs, *, end_str: str, model_ids, horizons=("1d", "5d", "20d")) -> list[pd.Series]:
    """Load full-window pred_<h>.pkl for the given model_ids at end_str from
    <model>_<h>_<end_str> recorders. Mirrors run_split's pooling load."""
    out: dict[str, pd.Series] = {}
    for rec in recs:
        info = getattr(rec, "info", {})
        run_name = info.get("name") if isinstance(info, dict) else getattr(rec, "name", "")
        for model_id in model_ids:
            for h in horizons:
                if run_name != f"{model_id}_{h}_{end_str}":
                    continue
                try:
                    obj = rec.load_object(f"pred_{h}.pkl")
                except Exception as exc:
                    _log.warning("reblend_load_failed recorder=%s file=pred_%s.pkl error=%s", run_name, h, exc)
                    continue
                s = (obj["score"] if "score" in obj.columns else obj.iloc[:, 0]) if isinstance(obj, pd.DataFrame) else obj
                col = f"{model_id}_{h}"
                if col not in out:
                    out[col] = s.rename(col)
    return list(out.values())
