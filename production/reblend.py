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


def candidate_experiment_name(prod_experiment: str) -> str:
    return f"{prod_experiment}_candidates"


def latest_full_fold_end_date(cfg):
    """Return the newest ensemble_<date> recorder's date in the production experiment."""
    from datetime import date as _date
    from qlib.workflow import R
    recs = R.list_recorders(experiment_name=cfg.experiment_name)
    dates = []
    for rec in recs.values():
        nm = (rec.info or {}).get("name", "") if isinstance(rec.info, dict) else getattr(rec, "name", "")
        if nm.startswith("ensemble_"):
            try:
                dates.append(_date.fromisoformat(nm[len("ensemble_"):]))
            except ValueError:
                pass
    if not dates:
        raise RuntimeError("no ensemble_<date> recorder found — run a full retrain first")
    return max(dates)


def reblend_single(cfg, target_model_id: str, end_date):
    """Retrain target_model_id on end_date, reuse the other enabled models' cached
    preds, and write a candidate ensemble recorder. Returns the candidate recorder_name."""
    from datetime import date as _date
    from qlib.workflow import R
    from production.rolling_train import (
        build_universe, init_qlib, train_lgbm_horizon, pool_stack_write,
    )
    if isinstance(end_date, str):
        end_date = _date.fromisoformat(end_date)
    init_qlib(cfg)
    members, universe_name = build_universe(cfg, end_date)

    enabled = [s["id"] for s in cfg.model_specs if s.get("enabled")]
    if target_model_id not in enabled:
        raise ValueError(f"{target_model_id} not in enabled models {enabled}")
    others = [m for m in enabled if m != target_model_id]

    series_list = []
    if target_model_id == "lgbm":
        for h in cfg.horizons:
            series_list.append(train_lgbm_horizon(cfg, h, universe_name, end_date))
    elif target_model_id == "alstm":
        from production.train_alstm import train_alstm_multihead
        series_list.extend(train_alstm_multihead(cfg, universe_name, end_date))
    elif target_model_id == "tra":
        from production.train_tra import train_tra_multihead
        series_list.extend(train_tra_multihead(cfg, universe_name, end_date))
    else:
        raise ValueError(f"unknown model {target_model_id}")

    recs = list(R.list_recorders(experiment_name=cfg.experiment_name).values())
    reused = series_from_recorders(recs, end_str=end_date.isoformat(), model_ids=tuple(others))
    if others and not reused:
        raise RuntimeError(
            f"no cached recorders for {others} at end_date={end_date} — run a full retrain for this fold first"
        )
    series_list.extend(reused)

    cand_exp = candidate_experiment_name(cfg.experiment_name)
    recorder_name = f"candidate_{target_model_id}_{end_date.isoformat()}"
    pool_stack_write(cfg, series_list, end_date, members,
                     experiment_name=cand_exp, recorder_name=recorder_name, seed_serving=True)
    _log.info("reblend_candidate_written recorder=%s exp=%s", recorder_name, cand_exp)
    return recorder_name
