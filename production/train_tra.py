"""TRA multi-head training wrapper.

Like train_alstm.py but uses qlib's TRA model. TRA is internally multi-task
(K=3 states with optimal-transport routing per stock per day) but its
*label* output is single-dim, so we run it once per horizon and emit
tra_1d / tra_5d / tra_20d series — same β simplification as ALSTM.

Mid-week training fix
---------------------
Qlib's `TRAModel.test_epoch` aggregates per-batch metrics at the end of the
loop with `pd.DataFrame(metrics_list).MSE.mean()`. If the test segment's
labels are partially or fully NaN (the freshest 5d/20d window before
realised labels arrive), the per-batch `evaluate()` calls either return
dicts without an `MSE` key (because `pred.rank(pct=True)` collapses) or
raise inside `transport_fn` during pretrain's "oracle" path. The
aggregation then dies with `'DataFrame' object has no attribute 'MSE'`,
which propagates up through `model.fit()` and `model.predict()` and TRA
gets skipped for that horizon.

We monkey-patch `test_epoch` with a version that

  1. Wraps each per-batch `evaluate(pred)` call in try/except and substitutes
     NaN metrics when it raises.
  2. Wraps each `transport_fn` call too (oracle on NaN labels can blow up
     before even reaching `evaluate`); skips the batch on failure.
  3. Reads `MSE`/`MAE`/`IC` columns defensively from the aggregation
     DataFrame, returning NaN when any column is missing.

The predictions themselves are router-mode (no label dependency) so they
remain valid even when the metrics are NaN — exactly what we need to score
freshly trained models on the current week.
"""
from __future__ import annotations

import gc
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from production.train_alstm import _build_multihead_dataset

_log = logging.getLogger("train_tra")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _install_safe_tra_test_epoch() -> None:
    """Replace `TRAModel.test_epoch` with a NaN-tolerant variant.

    Idempotent: re-installing is a no-op.
    """
    import numpy as np
    import torch
    from tqdm import tqdm
    from qlib.contrib.model.pytorch_tra import TRAModel, evaluate

    if getattr(TRAModel, "_safe_test_epoch_installed", False):
        return

    def safe_test_epoch(self, epoch, data_set, return_pred=False, prefix="test", is_pretrain=False):  # type: ignore[no-redef]
        self.model.eval()
        self.tra.eval()
        data_set.eval()

        preds = []
        probs = []
        P_all = []
        metrics = []
        for batch in tqdm(data_set):
            data, state, label, count = batch["data"], batch["state"], batch["label"], batch["daily_count"]
            index = batch["daily_index"] if self.use_daily_transport else batch["index"]

            with torch.no_grad():
                hidden = self.model(data)
                all_preds, choice, prob = self.tra(hidden, state)

            try:
                if is_pretrain or self.transport_method != "none":
                    loss, pred, L, P = self.transport_fn(
                        all_preds,
                        label,
                        choice,
                        prob,
                        state.mean(dim=1),
                        count,
                        self.transport_method if not is_pretrain else "oracle",
                        self.alpha,
                        training=False,
                    )
                    data_set.assign_data(index, L)  # save loss to memory
                    if P is not None and return_pred:
                        P_all.append(pd.DataFrame(P.cpu().numpy(), index=index))
                else:
                    pred = all_preds.mean(dim=1)
            except Exception:
                # transport_fn can crash on all-NaN labels (oracle path); fall
                # back to the router/no-state baseline so we still emit a pred.
                pred = all_preds.mean(dim=1)

            X = np.c_[pred.cpu().numpy(), label.cpu().numpy(), all_preds.cpu().numpy()]
            columns = ["score", "label"] + ["score_%d" % d for d in range(all_preds.shape[1])]
            pred = pd.DataFrame(X, index=batch["index"], columns=columns)

            try:
                m = evaluate(pred)
            except Exception:
                m = {"MSE": float("nan"), "MAE": float("nan"), "IC": float("nan")}
            metrics.append(m)

            if return_pred:
                preds.append(pred)
                if prob is not None:
                    columns = ["prob_%d" % d for d in range(all_preds.shape[1])]
                    probs.append(pd.DataFrame(prob.cpu().numpy(), index=index, columns=columns))

        # Defensive aggregation — empty list, or rows without MSE/MAE/IC keys
        # (which is what crashes qlib's original test_epoch).
        agg = pd.DataFrame(metrics) if metrics else pd.DataFrame()
        def _safe_mean(col: str) -> float:
            if col not in agg.columns or len(agg) == 0:
                return float("nan")
            v = agg[col].mean()
            return float("nan") if pd.isna(v) else float(v)
        ic_mean = _safe_mean("IC")
        ic_std = float("nan")
        if "IC" in agg.columns and len(agg) > 1:
            s = agg["IC"].std()
            ic_std = float("nan") if pd.isna(s) or s == 0 else float(s)
        icir = ic_mean / ic_std if (ic_std and not pd.isna(ic_std)) else float("nan")
        metrics = {
            "MSE": _safe_mean("MSE"),
            "MAE": _safe_mean("MAE"),
            "IC": ic_mean,
            "ICIR": icir,
        }

        if self._writer is not None and epoch >= 0 and not is_pretrain:
            for key, value in metrics.items():
                if not pd.isna(value):
                    self._writer.add_scalar(prefix + "/" + key, value, epoch)

        if return_pred:
            preds = pd.concat(preds, axis=0) if preds else pd.DataFrame(columns=["score", "label"])
            probs = pd.concat(probs, axis=0) if probs else None
            P_all = pd.concat(P_all, axis=0) if P_all else None
            return metrics, preds, probs, P_all
        return metrics

    TRAModel.test_epoch = safe_test_epoch
    TRAModel._safe_test_epoch_installed = True
    _log.info("safe_tra_test_epoch_installed — NaN-label tolerant")


def _load_tra_config(cfg=None) -> dict:
    """Load the TRA YAML pointed at by cfg.model_specs (so smoke configs
    work via --config). Falls back to the production default if no cfg is
    given (back-compat for tests)."""
    if cfg is not None:
        rel = next(m["config"] for m in cfg.model_specs if m["id"] == "tra")
        path = REPO_ROOT / rel
    else:
        path = REPO_ROOT / "production/configs/tra_alpha360.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_tra_multihead(cfg, universe_name: str, end_date: date) -> list[pd.Series]:
    """Train TRA per horizon; return tra_1d, tra_5d, tra_20d Series.

    Failure is non-fatal: if any horizon's training raises (e.g., GPU OOM,
    transport-method convergence failure), that horizon is logged and skipped.
    The remaining horizons still produce series, and the upstream ensemble
    degrades gracefully via rank-average over fewer columns.
    """
    from qlib.contrib.data.dataset import MTSDatasetH
    from qlib.contrib.model.pytorch_tra import TRAModel
    from qlib.workflow import R

    # Patch qlib's TRAModel.test_epoch so mid-week training (NaN test labels
    # for the freshest 5d/20d windows) doesn't crash on metric aggregation.
    _install_safe_tra_test_epoch()

    tra_yaml = _load_tra_config(cfg)
    mhd = _build_multihead_dataset(cfg, universe_name, end_date)
    # num_states must match tra_config.num_states from the YAML so the
    # OT routing receives the same K it was configured for.
    num_states = int(tra_yaml["model"]["kwargs"]["tra_config"]["num_states"])
    # memory_mode must match between the dataset (which allocates the
    # per-sample / per-day memory matrix) and the TRAModel (which decides
    # whether to use daily OT transport). 'sample' is the canonical default.
    memory_mode = tra_yaml["model"]["kwargs"].get("memory_mode", "sample")
    # `input_size` reshapes Alpha360's flat 360 columns into (60 days × 6
    # features) per sample (see MTSDatasetH __iter__:
    # `self._data[slc.stop - 1].reshape(self.input_size, -1).T`). When
    # `input_size` is supplied, the dataset bypasses its normal seq_len
    # padding and instead trusts the 360 cols already encode the lookback,
    # so seq_len becomes a no-op for slicing but is kept at the canonical
    # 60 for downstream code that consults `dataset.seq_len` (e.g. TRA
    # memory/state padding). Matches `seq_len: 60`, `input_size: 6` in
    # qlib's canonical `workflow_config_tra_Alpha360.yaml`.
    input_size = int(tra_yaml["model"]["kwargs"]["model_config"]["input_size"])
    outputs: list[pd.Series] = []
    for h in cfg.horizons:
        # Same lazy-handler pattern as train_alstm: build, train, free, gc.
        handler = mhd.build_handler(h.name)
        # TRAModel hard-rejects anything that isn't an MTSDatasetH; it relies
        # on the sampler's batch shape (N×T×D) and per-sample state vectors.
        dataset = MTSDatasetH(
            handler=handler,
            segments={
                "train": mhd.train_segment,
                "valid": mhd.valid_segment,
                "test": mhd.test_segment,
            },
            seq_len=60,
            horizon=cfg.horizon_days[h.name],
            num_states=num_states,
            memory_mode=memory_mode,
            batch_size=1024,
            n_samples=None,
            drop_last=True,        # canonical for training; dataset.eval() toggles to False at predict time
            input_size=input_size,
        )
        model = TRAModel(**tra_yaml["model"]["kwargs"])
        with R.start(experiment_name=cfg.experiment_name, recorder_name=f"tra_{h.name}_{end_date}"):
            try:
                model.fit(dataset)
                pred = model.predict(dataset)
                R.save_objects(**{f"pred_{h.name}.pkl": pred})
                # TRA.predict() returns a DataFrame with columns
                # ['score', 'label', 'score_0'..'score_K-1'] (per-state preds).
                # The 'score' col is the OT-routed prediction we want.
                # See qlib.contrib.model.pytorch_tra.TRAModel.test_epoch (line ~309).
                if isinstance(pred, pd.DataFrame):
                    if "score" in pred.columns:
                        pred_series = pred["score"]
                    else:
                        pred_series = pred.iloc[:, 0]
                else:
                    pred_series = pred
                outputs.append(pred_series.rename(f"tra_{h.name}"))
            except Exception as exc:
                _log.warning("tra_failed_skipping horizon=%s error=%s", h.name, str(exc))
        del handler, dataset, model
        gc.collect()

    return outputs
