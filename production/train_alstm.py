"""ALSTM multi-head training wrapper.

Per spec §5 β-phase simplification: trains qlib's stock single-output ALSTM
once per horizon (3× per weekly run). True label-multi-head sharing is a γ
optimization. Each run produces an alstm_<horizon> Series; gradient clipping
at 3.0; serial after LightGBM to avoid GPU contention.
"""
from __future__ import annotations

import gc
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
    """Lazy multi-head dataset descriptor.

    Holds the shared segment tuple but does NOT pre-build all 3 Alpha360
    handlers up front (that would cost ~3x the per-handler memory, which is
    the source of the joblib MaybeEncodingError/MemoryError when running
    csi800 with 5-year windows). Callers build one handler per horizon via
    `build_handler(horizon_name)` and release it before the next.
    """

    build_handler: callable  # (horizon_name: str) -> handler instance
    label_cols: list[str]
    train_segment: tuple[str, str]
    valid_segment: tuple[str, str]
    test_segment: tuple[str, str]


def _build_multihead_dataset(
    cfg, universe_name: str, end_date: date, build_features: bool = True
) -> MultiHeadDataset:
    """Return a descriptor with a per-horizon handler factory (lazy build).

    β simplification: all 3 handlers use the 5d horizon's time window. The 1d
    handler thus gets a 5-year training window (not 3 per its own config) and
    the 20d handler also gets 5 years (not 7).

    The handlers are built one-at-a-time by the caller (via the returned
    `build_handler` callable) so only one Alpha360 frame (~1-2 GB for the
    csi800 × 5y window) lives in memory at any given time. Pre-building all
    3 handlers up front previously triggered joblib MemoryError during the
    second handler's parallel feature load.

    See spec §5 and §6 for the trade-off rationale.
    """
    prod_path = str((REPO_ROOT / "production").resolve())
    if prod_path not in sys.path:
        sys.path.insert(0, prod_path)

    h5 = next(h for h in cfg.horizons if h.name == "5d")
    s = split(end_date=end_date, cfg=h5)

    if build_features:
        from custom_handler import Alpha360_OpenH

        # Load processors from the ALSTM YAML once per call (closed over by
        # the factory below).
        model_cfg_path = REPO_ROOT / [
            m for m in cfg.model_specs if m["id"] == "alstm"
        ][0]["config"]
        with model_cfg_path.open(encoding="utf-8") as f:
            alstm_yaml = yaml.safe_load(f)
        learn_procs = alstm_yaml.get("learn_processors", [])
        infer_procs = alstm_yaml.get("infer_processors", [])

        def _factory(horizon_name: str):
            return Alpha360_OpenH(
                horizon_days=cfg.horizon_days[horizon_name],
                start_time=str(s.train_start),
                end_time=str(s.test_end),
                fit_start_time=str(s.train_start),
                fit_end_time=str(s.train_end),
                instruments=universe_name,
                learn_processors=learn_procs,
                infer_processors=infer_procs,
            )
    else:
        def _factory(horizon_name: str):
            return None

    return MultiHeadDataset(
        build_handler=_factory,
        label_cols=[f"LABEL_{h.name}" for h in cfg.horizons],
        train_segment=(str(s.train_start), str(s.train_label_end)),
        valid_segment=(str(s.valid_start), str(s.valid_end)),
        test_segment=(str(s.test_start), str(s.test_end)),
    )


def train_alstm_multihead(cfg, universe_name: str, end_date: date) -> list[pd.Series]:
    """Train ALSTM per horizon (β simplification — see spec §5);
    return 3 prediction Series named alstm_1d / _5d / _20d.

    Uses qlib's canonical Alpha360 ALSTM config (see
    `examples/benchmarks/ALSTM/workflow_config_alstm_Alpha360.yaml`):
    `pytorch_alstm.ALSTM` (NOT `_ts`) + plain `DatasetH`. The non-`_ts`
    model internally reshapes the flat 360-col Alpha360 row via
    `inputs.view(batch, d_feat=6, -1)` so the 60-day lookback dimension is
    recovered. The `_ts` variant requires already-time-formatted data from
    `TSDatasetH`, which would produce shape `(batch, step_len, 360)` and
    fail with `d_feat=6` (mat1 360 x mat2 6x64 mismatch).

    TODO(γ): explicit gradient clipping currently relies on
    `pytorch_alstm.ALSTM`'s hard-coded `clip_grad_value_(..., 3.0)`. To
    honor `grad_clip_max_norm` from YAML we need to override the train
    loop in γ phase.
    """
    from qlib.contrib.model.pytorch_alstm import ALSTM
    from qlib.data.dataset import DatasetH
    from qlib.workflow import R

    model_cfg_path = REPO_ROOT / [m for m in cfg.model_specs if m["id"] == "alstm"][0]["config"]
    with model_cfg_path.open(encoding="utf-8") as f:
        alstm_yaml = yaml.safe_load(f)

    mhd = _build_multihead_dataset(cfg, universe_name, end_date)
    outputs: list[pd.Series] = []
    # `pytorch_alstm.ALSTM` doesn't accept `n_jobs` (only `pytorch_alstm_ts`
    # does). Strip it from kwargs in case the YAML still carries it.
    model_kwargs = {k: v for k, v in alstm_yaml["model"]["kwargs"].items() if k != "n_jobs"}
    for h in cfg.horizons:
        # Build a fresh handler for this horizon only; we free it at the end
        # of the iteration so the next horizon's Alpha360 frame is not
        # accumulating with the previous one's (avoids joblib MemoryError).
        handler = mhd.build_handler(h.name)
        dataset = DatasetH(
            handler=handler,
            segments={
                "train": mhd.train_segment,
                "valid": mhd.valid_segment,
                "test": mhd.test_segment,
            },
        )
        model = ALSTM(**model_kwargs)
        with R.start(experiment_name=cfg.experiment_name, recorder_name=f"alstm_{h.name}_{end_date}"):
            try:
                model.fit(dataset)
                pred = model.predict(dataset)
                R.save_objects(**{f"pred_{h.name}.pkl": pred})
                outputs.append(pred.rename(f"alstm_{h.name}"))
            except Exception as exc:
                _log.warning("alstm_failed_skipping horizon=%s error=%s", h.name, str(exc))
        # Explicitly drop references to the per-horizon handler + dataset +
        # model so their Alpha360 frames are collectible before the next
        # horizon's handler load.
        del handler, dataset, model
        gc.collect()

    return outputs
