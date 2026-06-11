"""Event-driven daily inference.

Triggered by the data refresh job's success callback (or manually by
`POST /api/inference/run-now`). Loads the latest <model>_<horizon>_<date>
weekly recorders, runs inference on missing dates, applies isotonic
calibration, appends new rows to the pooled ensemble recorder's pred.pkl.

Usage:
  python -m production.daily_inference
  python -m production.daily_inference --end-date 2026-05-27
  python -m production.daily_inference --force

Design references:
  - docs/superpowers/specs/2026-05-28-prediction-ux-redesign-design.md §5
  - docs/superpowers/plans/2026-05-28-prediction-ux-redesign.md Task 5
"""
from __future__ import annotations

import sys as _sys
import sysconfig as _sysconfig
_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import argparse
import copy
import json
import logging
import time
from datetime import date, datetime
from pathlib import Path

import urllib.request

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(REPO_ROOT))

from production.calibration import apply_calibration, load_calibration

log = logging.getLogger("daily_inference")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

HORIZONS = ("1d", "5d", "20d")
MODELS = ("lgbm", "alstm", "tra")
CACHE_PATH = REPO_ROOT / "production" / "cache" / "latest_calibration.pkl"
INVALIDATE_URL = "http://127.0.0.1:8000/api/internal/cache/invalidate"
ANALYSIS_REFRESH_URL = "http://127.0.0.1:8000/api/internal/analysis/refresh"


# ---- helpers (pure, easy to test) -----------------------------------------

def _recorder_name(rec) -> str:
    info = rec.info if hasattr(rec, "info") else {}
    name = info.get("name") if isinstance(info, dict) else getattr(rec, "name", "")
    if name:
        return name
    try:
        return rec.client.get_run(rec.id).data.tags.get("mlflow.runName", "")
    except Exception:
        return ""


def _handler_signature(cfg: dict) -> str:
    """Two configs are 'same handler' if class + kwargs (minus segment-specific
    fields) match.

    daily_inference overrides start_time / end_time / instruments at infer time,
    so those don't need to be part of the equality key. Including them would
    falsely split otherwise-identical handlers.
    """
    cls = cfg.get("class", "?")
    kw = {k: v for k, v in cfg.get("kwargs", {}).items()
          if k not in ("start_time", "end_time", "fit_start_time", "fit_end_time",
                       "instruments")}
    return cls + ":" + json.dumps(kw, sort_keys=True, default=str)


def _default_handler_cfg(model_id: str) -> dict:
    """Fallback config when handler_config.pkl artifact is absent on an
    older recorder."""
    if model_id == "lgbm":
        return {
            "class": "Alpha158_OpenH",
            "module_path": "custom_handler",
            "kwargs": {},
        }
    return {
        "class": "Alpha360_OpenH",
        "module_path": "custom_handler",
        "kwargs": {},
    }


def _missing_dates(qlib_dates, pred_dates) -> list[date]:
    """Set difference: dates in qlib calendar but not in pred.pkl."""
    q = {d.date() if hasattr(d, "date") else d for d in qlib_dates}
    p = {d.date() if hasattr(d, "date") else d for d in pred_dates}
    return sorted(q - p)


def _group_by_handler_signature(loaded: dict) -> dict[str, list]:
    """{model_id: {horizon: (model, cfg)}}  ->  {sig: [(mid, h, model, cfg), ...]}

    Used to avoid building the same Alpha360 feature frame 3x for ALSTM and
    3x for TRA — one shared feature build per handler signature.
    """
    groups: dict[str, list] = {}
    for mid, hmap in loaded.items():
        for h, (model, cfg) in hmap.items():
            sig = _handler_signature(cfg)
            groups.setdefault(sig, []).append((mid, h, model, cfg))
    return groups


def _composite_and_calibrate(raw: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """Given a DataFrame whose columns include some/all of
    {lgbm/alstm/tra}_{1d/5d/20d}, compute:

      - composite_<h>           rank-avg per horizon
      - expected_return_<h>     if cal[h] available
      - score                   v9 convention: -rank_avg of 1d+5d cols
      - consensus               3-model directional agreement on 5d

    Returns the enriched DataFrame.
    """
    df = raw.copy()
    cal_maps = cal.get("maps", {}) if isinstance(cal, dict) else {}

    for h in HORIZONS:
        cols = [c for c in df.columns
                if c.endswith(f"_{h}")
                and not c.startswith("expected_")
                and not c.startswith("composite_")]
        if not cols:
            continue
        ranks = df[cols].groupby(level="datetime").rank(ascending=False, method="min")
        comp = -ranks.mean(axis=1, skipna=True)
        df[f"composite_{h}"] = comp
        if h in cal_maps:
            df[f"expected_return_{h}"] = apply_calibration(comp, cal_maps[h])

    # Unified score = -rank_avg over 1d+5d cols (exclude 20d per v9)
    score_cols = [c for c in df.columns
                  if (c.endswith("_1d") or c.endswith("_5d"))
                  and not (c.startswith("expected_") or c.startswith("composite_"))]
    if score_cols:
        ranks = df[score_cols].groupby(level="datetime").rank(ascending=False, method="min")
        df["score"] = -ranks.mean(axis=1, skipna=True)

    # Consensus on 5d direction
    five_cols = [c for c in df.columns
                 if c.endswith("_5d")
                 and not (c.startswith("expected_") or c.startswith("composite_"))]
    if five_cols:
        five = df[five_cols]
        signs = np.sign(five.fillna(0).values)
        # consensus = max(|sum_pos|, |sum_neg|) / n_non_nan
        n_models = signs.shape[1]
        net = signs.sum(axis=1)
        df["consensus"] = np.abs(net) / n_models

    return df


# ---- pipeline ----------------------------------------------------------------

def _load_models(exp_name: str) -> dict:
    """Return {model_id: {horizon: (model_obj, handler_cfg)}}"""
    from qlib.workflow import R
    exp = R.get_exp(experiment_name=exp_name)
    recs = exp.list_recorders()
    if isinstance(recs, dict):
        recs = list(recs.values())

    out: dict = {}
    for mid in MODELS:
        out[mid] = {}
        for h in HORIZONS:
            prefix = f"{mid}_{h}_"
            matched = [r for r in recs if _recorder_name(r).startswith(prefix)]
            if not matched:
                log.warning("no_recorder model=%s horizon=%s", mid, h)
                continue
            latest = max(matched, key=lambda r: r.info.get("start_time", 0))
            try:
                model = latest.load_object("trained_model")
            except Exception as exc:
                log.warning("trained_model_load_failed model=%s horizon=%s: %s",
                            mid, h, exc)
                continue
            try:
                cfg = latest.load_object("handler_config.pkl")
            except Exception:
                cfg = _default_handler_cfg(mid)
                log.warning("handler_config_missing model=%s horizon=%s fallback=default",
                            mid, h)
            out[mid][h] = (model, cfg)
            log.info("loaded model=%s horizon=%s recorder=%s",
                     mid, h, _recorder_name(latest)[:50])
    return out


def _find_pooled_recorder(exp_name: str):
    """The latest 'ensemble_'-prefixed recorder is what the backend serves."""
    from qlib.workflow import R
    exp = R.get_exp(experiment_name=exp_name)
    recs = exp.list_recorders()
    if isinstance(recs, dict):
        recs = list(recs.values())
    pooled = [r for r in recs if _recorder_name(r).startswith("ensemble_")]
    if not pooled:
        raise RuntimeError(f"no pooled recorder found in {exp_name}")
    return max(pooled, key=lambda r: r.info.get("start_time", 0))


def _infer_group(group, dates, instruments) -> dict[str, pd.Series]:
    """Build feature dataset once per handler group, predict each model."""
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset import DatasetH
    # Ensure custom_handler module path is available
    prod_path = str((REPO_ROOT / "production").resolve())
    if prod_path not in sys.path:
        sys.path.insert(0, prod_path)

    cfg = copy.deepcopy(group[0][3])
    cfg.setdefault("kwargs", {})
    cfg["kwargs"].update(
        start_time=str(dates[0]),
        end_time=str(dates[-1]),
        instruments=instruments,
    )
    handler = init_instance_by_config(cfg)
    dataset = DatasetH(
        handler=handler,
        segments={"test": (str(dates[0]), str(dates[-1]))},
    )
    out: dict[str, pd.Series] = {}
    for mid, h, model, _ in group:
        try:
            pred = model.predict(dataset)
            if isinstance(pred, pd.DataFrame):
                pred = pred["score"] if "score" in pred.columns else pred.iloc[:, 0]
            out[f"{mid}_{h}"] = pred
        except Exception as exc:
            log.warning("predict_failed model=%s horizon=%s: %s", mid, h, exc)
    return out


def _post_invalidate_cache():
    """Best-effort: notify backend the candidates cache is stale."""
    import urllib.request
    try:
        req = urllib.request.Request(INVALIDATE_URL, method="POST")
        with urllib.request.urlopen(req, timeout=3) as r:
            log.info("cache_invalidate status=%d", r.status)
    except Exception as exc:
        log.warning("cache_invalidate_failed: %s — backend may be down", exc)


def _post_analysis_refresh():
    """Best-effort: ask the backend to (re)generate AI analysis for the top-N picks.
    No-op server-side when ai_analysis_enabled is false / no key."""
    try:
        req = urllib.request.Request(ANALYSIS_REFRESH_URL, method="POST")
        with urllib.request.urlopen(req, timeout=3) as r:
            log.info("analysis_refresh status=%d", r.status)
    except Exception as exc:
        log.warning("analysis_refresh_failed: %s — backend may be down", exc)


def run(end_date: date | None = None, force: bool = False,
        experiment: str = "rolling_v2_ensemble") -> int:
    """Main pipeline. Returns exit code (0=success)."""
    from qlib.data import D
    from production.rolling_train import load_config, init_qlib

    cfg = load_config(REPO_ROOT / "production/configs/rolling_ensemble.yaml")
    init_qlib(cfg)

    pooled = _find_pooled_recorder(experiment)
    try:
        existing = pooled.load_object("pred.pkl")
    except Exception as exc:
        log.error("pooled_pred_load_failed: %s", exc)
        return 1
    if not isinstance(existing, pd.DataFrame):
        existing = pd.DataFrame(existing) if hasattr(existing, "to_frame") else None
    if existing is None or existing.empty:
        log.error("pooled_pred_empty")
        return 1

    pred_dates = set(existing.index.get_level_values("datetime").unique())

    # qlib calendar — find latest available date
    cal_dates = D.calendar(end_time=str(end_date) if end_date else None)
    if len(cal_dates) == 0:
        log.error("empty_calendar")
        return 1
    qlib_latest = pd.Timestamp(cal_dates[-1])

    if force:
        # In force mode, recompute last 10 days even if already in pred
        missing = sorted({pd.Timestamp(d).date() for d in cal_dates[-10:]})
    else:
        # Only the dates after the most recent pred date, within last 30 days
        max_pred = max(pred_dates) if pred_dates else None
        if max_pred is not None:
            target_cal = [d for d in cal_dates if pd.Timestamp(d) > pd.Timestamp(max_pred)]
        else:
            target_cal = list(cal_dates[-30:])
        missing = _missing_dates(target_cal, pred_dates)

    if not missing:
        log.info("no_missing_dates pred_latest=%s qlib_latest=%s",
                 max(pred_dates) if pred_dates else None, qlib_latest)
        return 0

    log.info("missing_dates count=%d range=%s..%s",
             len(missing), missing[0], missing[-1])

    loaded = _load_models(experiment)
    if not any(loaded.values()):
        # All recorders lack the 'trained_model' artifact (pre-Task-3 training).
        # This is recoverable: the next weekly retrain will save the artifacts
        # via train_helpers.save_calibration_artifacts. Until then we exit
        # cleanly so the refresh callback doesn't appear to be failing.
        log.warning(
            "no_models_loaded — none of the weekly recorders have a "
            "'trained_model' artifact. This is expected for recorders trained "
            "before train_helpers.save_calibration_artifacts was added. "
            "The next weekly retrain will populate the artifacts and "
            "daily_inference will produce new predictions from then on. "
            "Until then, the existing pred.pkl is served unchanged."
        )
        return 0

    groups = _group_by_handler_signature(loaded)
    log.info("handler_groups count=%d sizes=%s",
             len(groups), [len(g) for g in groups.values()])

    instruments = sorted({inst for _, inst in existing.index})
    raw_scores: dict[str, pd.Series] = {}
    for sig, group in groups.items():
        log.info("group_start sig=%s n=%d", sig[:60], len(group))
        t0 = time.time()
        out = _infer_group(group, missing, instruments)
        log.info("group_end elapsed=%.1fs cols=%s",
                 time.time() - t0, list(out.keys()))
        raw_scores.update(out)

    if not raw_scores:
        log.error("no_predictions_produced")
        return 1

    raw_df = pd.concat(list(raw_scores.values()), axis=1).sort_index()
    raw_df = raw_df[~raw_df.index.duplicated(keep="last")]
    raw_df.index.names = ["datetime", "instrument"]

    # Only keep rows in the missing dates (Alpha360 handler may produce extra
    # leading rows for lookback; we slice to what's truly new).
    miss_ts = {pd.Timestamp(d) for d in missing}
    raw_df = raw_df.loc[raw_df.index.get_level_values("datetime").isin(miss_ts)]

    cal = load_calibration(CACHE_PATH)
    enriched = _composite_and_calibrate(raw_df, cal)

    # Align columns with existing pooled pred.pkl. Keep existing cols + new
    # expected_return_<h> / composite_<h> cols. Cols in existing but not in
    # enriched stay NaN for the new rows.
    final_cols = list(existing.columns)
    for c in enriched.columns:
        if c not in final_cols:
            final_cols.append(c)
    enriched = enriched.reindex(columns=final_cols)

    combined = pd.concat([existing, enriched], axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    pooled.save_objects(**{"pred.pkl": combined})
    log.info("appended new_rows=%d total_rows=%d new_dates=%d",
             len(enriched), len(combined), len(missing))

    _post_invalidate_cache()
    _post_analysis_refresh()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--end-date", default=None,
                        help="YYYY-MM-DD; default = latest qlib date")
    parser.add_argument("--force", action="store_true",
                        help="Re-infer even if pred.pkl has the dates")
    parser.add_argument("--experiment", default="rolling_v2_ensemble")
    args = parser.parse_args()

    end = date.fromisoformat(args.end_date) if args.end_date else None
    return run(end_date=end, force=args.force, experiment=args.experiment)


if __name__ == "__main__":
    sys.exit(main())
