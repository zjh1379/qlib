"""TRA multi-head training wrapper.

Like train_alstm.py but uses qlib's TRA model. TRA is internally multi-task
(K=10 states with optimal-transport routing per stock per day) but its
*label* output is single-dim, so we run it once per horizon and emit
tra_1d / tra_5d / tra_20d series — same β simplification as ALSTM.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from production.train_alstm import _build_multihead_dataset

_log = logging.getLogger("train_tra")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_tra_config() -> dict:
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

    tra_yaml = _load_tra_config()
    mhd = _build_multihead_dataset(cfg, universe_name, end_date)
    # num_states must match tra_config.num_states from the YAML so the
    # OT routing receives the same K it was configured for.
    num_states = int(tra_yaml["model"]["kwargs"]["tra_config"]["num_states"])
    outputs: list[pd.Series] = []
    for h in cfg.horizons:
        handler = mhd.handler_objs[h.name]
        # TRAModel hard-rejects anything that isn't an MTSDatasetH; it relies
        # on the sampler's batch shape (N×T×D) and per-sample state vectors.
        # seq_len=20 matches `step_len: 20` in the alstm/tra YAMLs;
        # horizon comes from the current iteration's horizon_days entry.
        dataset = MTSDatasetH(
            handler=handler,
            segments={
                "train": mhd.train_segment,
                "valid": mhd.valid_segment,
                "test": mhd.test_segment,
            },
            seq_len=20,
            horizon=cfg.horizon_days[h.name],
            num_states=num_states,
            batch_size=1024,
            n_samples=None,
            drop_last=False,
        )
        model = TRAModel(**tra_yaml["model"]["kwargs"])
        with R.start(experiment_name=cfg.experiment_name, recorder_name=f"tra_{h.name}_{end_date}"):
            try:
                model.fit(dataset)
                pred = model.predict(dataset)
                R.save_objects(**{f"pred_{h.name}.pkl": pred})
                outputs.append(pred.rename(f"tra_{h.name}"))
            except Exception as exc:
                _log.warning("tra_failed_skipping horizon=%s error=%s", h.name, str(exc))

    return outputs
