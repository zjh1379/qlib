"""TRA multi-head training wrapper.

Like train_alstm.py but uses qlib's TRA model. TRA is internally multi-task
(K=10 states with optimal-transport routing per stock per day) but its
*label* output is single-dim, so we run it once per horizon and emit
tra_1d / tra_5d / tra_20d series — same β simplification as ALSTM.
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
