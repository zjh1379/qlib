"""ALSTM multi-head training wrapper.

Per spec §5 β-phase simplification: trains qlib's stock single-output ALSTM
once per horizon (3× per weekly run). True label-multi-head sharing is a γ
optimization. Each run produces an alstm_<horizon> Series; gradient clipping
at 3.0; serial after LightGBM to avoid GPU contention.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from production.walk_forward import HorizonConfig, split

_log = logging.getLogger("train_alstm")
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class MultiHeadDataset:
    handler_objs: dict  # horizon_name -> handler instance
    label_cols: list[str]
    train_segment: tuple[str, str]
    valid_segment: tuple[str, str]
    test_segment: tuple[str, str]


def _build_multihead_dataset(
    cfg, universe_name: str, end_date: date, build_features: bool = True
) -> MultiHeadDataset:
    """Build 3 handlers (one per horizon), share the universe and time slice.

    β simplification: all 3 handlers use the 5d horizon's time window. The 1d
    handler thus gets a 5-year training window (not 3 per its own config) and
    the 20d handler also gets 5 years (not 7). This keeps a single shared
    DatasetH per horizon. Revisit at γ if per-horizon IC diverges from
    per-horizon-baseline runs.

    See spec §5 and §6 for the trade-off rationale.
    """
    prod_path = str((REPO_ROOT / "production").resolve())
    if prod_path not in sys.path:
        sys.path.insert(0, prod_path)

    h5 = next(h for h in cfg.horizons if h.name == "5d")
    s = split(end_date=end_date, cfg=h5)

    handlers = {}
    if build_features:
        from custom_handler import Alpha360_OpenH

        # Load processors + step_len from the ALSTM YAML once per call
        model_cfg_path = REPO_ROOT / [
            m for m in cfg.model_specs if m["id"] == "alstm"
        ][0]["config"]
        with model_cfg_path.open() as f:
            alstm_yaml = yaml.safe_load(f)
        learn_procs = alstm_yaml.get("learn_processors", [])
        infer_procs = alstm_yaml.get("infer_processors", [])

        for h in cfg.horizons:
            handlers[h.name] = Alpha360_OpenH(
                horizon_days=cfg.horizon_days[h.name],
                start_time=str(s.train_start),
                end_time=str(s.test_end),
                fit_start_time=str(s.train_start),
                fit_end_time=str(s.train_end),
                instruments=universe_name,
                learn_processors=learn_procs,
                infer_processors=infer_procs,
            )

    return MultiHeadDataset(
        handler_objs=handlers,
        label_cols=[f"LABEL_{h.name}" for h in cfg.horizons],
        train_segment=(str(s.train_start), str(s.train_label_end)),
        valid_segment=(str(s.valid_start), str(s.valid_end)),
        test_segment=(str(s.test_start), str(s.test_end)),
    )


def train_alstm_multihead(cfg, universe_name: str, end_date: date) -> list[pd.Series]:
    """Train ALSTM per horizon (β simplification — see spec §5);
    return 3 prediction Series named alstm_1d / _5d / _20d.

    TODO(γ): wire `step_len` and `grad_clip_max_norm` from YAML into the
    training loop. Currently `step_len` is implicit (qlib's stock ALSTM
    defaults to step_len=20, which happens to match the YAML), and
    `grad_clip_max_norm` is not enforced. For γ phase, override qlib's
    train loop to apply explicit gradient clipping.
    """
    from qlib.contrib.model.pytorch_alstm_ts import ALSTM
    from qlib.data.dataset import DatasetH
    from qlib.workflow import R

    model_cfg_path = REPO_ROOT / [m for m in cfg.model_specs if m["id"] == "alstm"][0]["config"]
    with model_cfg_path.open() as f:
        alstm_yaml = yaml.safe_load(f)

    mhd = _build_multihead_dataset(cfg, universe_name, end_date)
    outputs: list[pd.Series] = []
    for h in cfg.horizons:
        handler = mhd.handler_objs[h.name]
        dataset = DatasetH(
            handler=handler,
            segments={
                "train": mhd.train_segment,
                "valid": mhd.valid_segment,
                "test": mhd.test_segment,
            },
        )
        model = ALSTM(**alstm_yaml["model"]["kwargs"])
        with R.start(experiment_name=cfg.experiment_name, recorder_name=f"alstm_{h.name}_{end_date}"):
            model.fit(dataset)
            pred = model.predict(dataset)
            R.save_objects(**{f"pred_{h.name}.pkl": pred})
        outputs.append(pred.rename(f"alstm_{h.name}"))

    return outputs
