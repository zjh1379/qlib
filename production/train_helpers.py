"""Shared helpers for the 3 per-model training entry points.

Currently exposes `save_calibration_artifacts()` which adds three extra
mlflow recorder artifacts needed by daily_inference + calibration:

  - valid_pred.pkl   model predictions on the validation slice
  - valid_label.pkl  realized labels on the validation slice
  - handler_config.pkl  the handler init dict so daily_inference can
                         rebuild features for new dates

All saves are fail-soft (log a warning + continue) so a problem here
doesn't bring down the whole training run.
"""
from __future__ import annotations

import logging

import pandas as pd

_log = logging.getLogger(__name__)

# Folds whose end_date is within this many days of "today" are treated as LIVE
# and seed the serving recorder; older (historical backfill) folds do not.
LIVE_SEED_MAX_AGE_DAYS = 10


def is_live_fold(end_date, *, today=None, max_age_days: int = LIVE_SEED_MAX_AGE_DAYS) -> bool:
    """True when `end_date` is recent enough that its pooled prediction should
    seed the serving recorder (backend get_latest_recorder_id / daily_inference).

    Historical backfill folds return False on purpose: get_latest_recorder_id and
    daily_inference._find_pooled_recorder order recorders by mlflow start_time, so
    a backfill's last-trained (newest start_time) fold would otherwise hijack
    serving with a STALE data date. Accepts a date or an ISO 'YYYY-MM-DD' string.
    """
    from datetime import date as _date
    end = end_date if isinstance(end_date, _date) else _date.fromisoformat(str(end_date)[:10])
    ref = today or _date.today()
    return 0 <= (ref - end).days <= max_age_days


def seed_serving_recorder(experiment_name: str, end_date, pred_df, *,
                          today=None, max_age_days: int = LIVE_SEED_MAX_AGE_DAYS,
                          _r=None) -> bool:
    """Save `pred_df` as the pred.pkl artifact of the `ensemble_<end_date>`
    recorder so the backend (get_latest_recorder_id) + daily_inference can
    serve/extend it. THIS is the link the automated pooling paths were missing
    (they wrote only the examples/mlruns/pred_<date>.pkl file), which froze live
    serving at the last manually-pooled recorder.

    No-op (returns False) for non-live folds (see is_live_fold) so historical
    backfill folds never hijack serving. Fail-soft: returns False on any error.
    `_r` injects a fake qlib.workflow.R for offline testing.
    """
    from datetime import date as _date
    end = end_date if isinstance(end_date, _date) else _date.fromisoformat(str(end_date)[:10])
    if not is_live_fold(end, today=today, max_age_days=max_age_days):
        return False
    try:
        if _r is None:
            from qlib.workflow import R as _r
        with _r.start(experiment_name=experiment_name, recorder_name=f"ensemble_{end}"):
            _r.save_objects(**{"pred.pkl": pred_df})
        return True
    except Exception as exc:
        _log.warning("seed_serving_recorder_failed end_date=%s error=%s", end, exc)
        return False


def save_calibration_artifacts(model, dataset, handler_cfg, *, recorder=None) -> None:
    """Save the 4 extra artifacts to the current mlflow recorder.

    Must be called inside an active `R.start(...)` block (or pass an explicit
    `recorder=...`). The current mlflow recorder is resolved via
    `qlib.workflow.R.save_objects` when `recorder` is None.

    Saves:
      - trained_model      the model object itself, needed by daily_inference
                           to run forward on new dates without retraining
      - valid_pred.pkl     model predictions on the validation slice
      - valid_label.pkl    realized labels on the validation slice
      - handler_config.pkl init config so daily_inference can rebuild features

    Parameters
    ----------
    model : qlib model instance with .predict(dataset, segment=...)
    dataset : qlib DatasetH instance with segments={'train','valid','test'}
    handler_cfg : dict with the handler init config. Pass None to skip.
    recorder : optional explicit recorder. If None, uses R.save_objects.
    """
    save_fn = _get_save_fn(recorder)

    # ---- trained_model -----------------------------------------------------
    # This is what daily_inference needs to run forward on new dates.
    try:
        save_fn(**{"trained_model": model})
    except Exception as exc:
        _log.warning("save_calibration_artifacts: trained_model failed: %s", exc)

    # ---- valid_pred.pkl --------------------------------------------------
    try:
        try:
            valid_pred = model.predict(dataset, segment="valid")
        except TypeError:
            # Some models don't accept `segment` kw; fall back to swapping
            # the dataset segments so test==valid and predicting normally.
            valid_pred = _predict_valid_via_segment_swap(model, dataset)
        if isinstance(valid_pred, pd.DataFrame):
            if "score" in valid_pred.columns:
                valid_pred = valid_pred["score"]
            else:
                valid_pred = valid_pred.iloc[:, 0]
        save_fn(**{"valid_pred.pkl": valid_pred})
    except Exception as exc:
        _log.warning("save_calibration_artifacts: valid_pred failed: %s", exc)

    # ---- valid_label.pkl -------------------------------------------------
    try:
        # qlib's DatasetH.prepare returns the raw label DataFrame for the
        # 'label' col_set when data_key='raw'.
        valid_label = dataset.prepare("valid", col_set="label", data_key="raw")
        if isinstance(valid_label, pd.DataFrame):
            valid_label = valid_label.iloc[:, 0]
        save_fn(**{"valid_label.pkl": valid_label})
    except Exception as exc:
        _log.warning("save_calibration_artifacts: valid_label failed: %s", exc)

    # ---- handler_config.pkl ---------------------------------------------
    if handler_cfg is not None:
        try:
            save_fn(**{"handler_config.pkl": handler_cfg})
        except Exception as exc:
            _log.warning("save_calibration_artifacts: handler_config failed: %s", exc)


def _get_save_fn(recorder):
    if recorder is not None:
        return recorder.save_objects
    from qlib.workflow import R
    return R.save_objects


def _predict_valid_via_segment_swap(model, dataset):
    """For models whose .predict() ignores the `segment` kwarg, build a
    transient dataset whose 'test' segment is the original 'valid' range,
    so model.predict(...) effectively returns the valid predictions.
    """
    import copy
    ds2 = copy.copy(dataset)
    if not hasattr(ds2, "segments"):
        raise RuntimeError("dataset has no .segments attribute")
    orig_test = ds2.segments.get("test")
    ds2.segments = dict(ds2.segments)
    ds2.segments["test"] = ds2.segments["valid"]
    try:
        return model.predict(ds2)
    finally:
        ds2.segments["test"] = orig_test
