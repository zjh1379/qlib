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
