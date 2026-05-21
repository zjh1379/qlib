"""Weekly rolling retrain entry point.

Usage:
  python -m production.rolling_train run-once [--end-date YYYY-MM-DD] [--config production/configs/rolling_ensemble.yaml]
  python -m production.rolling_train backfill 2024-01-01..2024-12-31
  python -m production.rolling_train evaluate <recorder-id>

Phase E (beta complete): runs all 3 base models (LightGBM/ALSTM/TRA) x 3 horizons,
fits a Ridge stacker on the valid-window OOF preds, falls back to rank-average
if stacking fails, and emits a single pred.pkl with score + consensus + base
columns. See docs/superpowers/specs/2026-05-21-rolling-ensemble-algorithm-design.md.

Known followups (deferred from beta to gamma):
    - ShadowTracker (production/shadow_tracker.py) is implemented and tested
      but not yet wired into run_once. The 4-week shadow-vs-prod IR comparison
      will never fire until this is integrated. Decision required: how to flag
      a recorder as 'shadow' -- via recorder_name prefix, a config flag, or
      a separate experiment name. See spec section 8 for the intended semantics.
    - Stacker OOF uses the valid window rather than a dedicated stack-fit
      window. Acceptable approximation since base models early-stop on valid.
    - PIT instruments file writes one range per stock (union of window),
      not true per-day membership. True per-day filtering happens upstream
      via the pit df reindex.
    - ALSTM/TRA train per-horizon, not multi-head -- see train_alstm.py and
      train_tra.py for the beta simplification rationale.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from production.consensus import consensus_per_row, write_pred_pkl
from production.pit_constituents import load_or_refresh, members_on
from production.post_process import ewma_smooth
from production.walk_forward import HorizonConfig, split

_log = logging.getLogger("rolling_train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RollingConfig:
    experiment_name: str
    universe: str
    provider_uri: str
    region: str
    horizons: list[HorizonConfig]
    horizon_days: dict[str, int]
    model_specs: list[dict[str, Any]]  # [{id, config, enabled}]
    ewma_alpha: float
    cost_bps: float
    archive_weeks: int


def load_config(path: Path) -> RollingConfig:
    with path.open() as f:
        raw = yaml.safe_load(f)
    horizons = []
    horizon_days = {}
    for h in raw["horizons"]:
        horizons.append(
            HorizonConfig(
                name=h["name"],
                train_years=h["train_years"],
                valid_years=h["valid_years"],
                stack_years=h["stack_years"],
                test_weeks=h["test_weeks"],
            )
        )
        horizon_days[h["name"]] = h["horizon_days"]
    return RollingConfig(
        experiment_name=raw["experiment_name"],
        universe=raw["universe"],
        provider_uri=raw["provider_uri"],
        region=raw["region"],
        horizons=horizons,
        horizon_days=horizon_days,
        model_specs=raw["models"],
        ewma_alpha=raw["post_process"]["ewma_alpha"],
        cost_bps=raw["post_process"]["cost_bps"],
        archive_weeks=raw["mlruns_archive"]["keep_weeks"],
    )


def init_qlib(cfg: RollingConfig) -> None:
    import qlib
    from qlib.constant import REG_CN, REG_US

    qlib.init(
        provider_uri=str(Path(cfg.provider_uri).expanduser()),
        region=REG_CN if cfg.region == "cn" else REG_US,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": f"file:{(REPO_ROOT / 'examples' / 'mlruns').resolve()}",
                "default_exp_name": cfg.experiment_name,
            },
        },
    )


def build_universe(cfg: RollingConfig, end_date: date) -> tuple[list[str], str]:
    """Build the PIT-correct training universe for `end_date`.

    Returns:
        members: list of qlib instrument codes (the universe of stocks the
                 handler should pull features for — used by callers for
                 reporting and for PIT-filtering training samples).
        universe_name: name of the qlib instruments file written to
                       `<provider_uri>/instruments/<universe_name>.txt`. Pass
                       this string (NOT the list) to handler `instruments=`.
    """
    from production.pit_constituents import write_pit_instruments_file

    pit = load_or_refresh(end=end_date)
    # Cross-section on end_date — useful for reporting and for the consensus
    # calc which only emits scores for stocks that were CSI800 members on the
    # test date.
    members = members_on(pit, end_date)

    universe_name = f"csi800_pit_{end_date.isoformat()}"
    qlib_data_root = Path(cfg.provider_uri).expanduser()
    # lookback_years should cover the longest train_years across horizons (20d = 7y)
    longest_lookback = max(h.train_years for h in cfg.horizons)
    write_pit_instruments_file(
        pit,
        end_date=end_date,
        name=universe_name,
        qlib_data_root=qlib_data_root,
        lookback_years=longest_lookback + 1,  # +1 for safety
    )
    return members, universe_name


def train_lgbm_horizon(
    cfg: RollingConfig,
    horizon: HorizonConfig,
    universe_name: str,
    end_date: date,
) -> pd.Series:
    """Train one LightGBM head and return its predictions on the test window.

    `universe_name` is the qlib instruments-file name produced by
    `build_universe()` — passed to the handler as a string so qlib resolves it
    via `D.instruments(market=universe_name)`.

    Returns a Series indexed by (datetime, instrument) named like 'lgbm_<horizon>'.
    """
    from qlib.contrib.data.handler import Alpha158  # noqa: F401  (registers ops)
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH
    from qlib.workflow import R

    s = split(end_date=end_date, cfg=horizon)
    _log.info(
        "horizon_split horizon=%s train=%s..%s valid=%s..%s test=%s..%s",
        horizon.name, s.train_start, s.train_end,
        s.valid_start, s.valid_end, s.test_start, s.test_end,
    )

    # Load LightGBM hyperparameters
    model_cfg_path = REPO_ROOT / [m for m in cfg.model_specs if m["id"] == "lgbm"][0]["config"]
    with model_cfg_path.open() as f:
        lgbm_yaml = yaml.safe_load(f)

    # Add production/ to sys.path so qlib can resolve custom_handler.Alpha158_OpenH
    prod_path = str((REPO_ROOT / "production").resolve())
    if prod_path not in sys.path:
        sys.path.insert(0, prod_path)

    from custom_handler import Alpha158_OpenH  # noqa: E402

    handler = Alpha158_OpenH(
        horizon_days=cfg.horizon_days[horizon.name],
        start_time=str(s.train_start),
        end_time=str(s.test_end),
        fit_start_time=str(s.train_start),
        fit_end_time=str(s.train_end),
        instruments=universe_name,
    )
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (str(s.train_start), str(s.train_label_end)),
            "valid": (str(s.valid_start), str(s.valid_end)),
            "test": (str(s.test_start), str(s.test_end)),
        },
    )

    model = LGBModel(**lgbm_yaml["model"]["kwargs"])
    with R.start(experiment_name=cfg.experiment_name, recorder_name=f"lgbm_{horizon.name}_{end_date}"):
        model.fit(dataset)
        pred = model.predict(dataset)
        R.save_objects(**{f"pred_{horizon.name}.pkl": pred})
    pred = pred.rename(f"lgbm_{horizon.name}")
    return pred


def run_once(cfg: RollingConfig, end_date: date) -> Path:
    """Run one weekly iteration. Returns the path to the written pred.pkl."""
    init_qlib(cfg)
    members, universe_name = build_universe(cfg, end_date)
    _log.info(
        "universe_built size=%d as_of=%s name=%s",
        len(members), str(end_date), universe_name,
    )

    # Train all enabled base models for all horizons
    series_list: list[pd.Series] = []
    for spec in cfg.model_specs:
        if not spec["enabled"]:
            continue
        if spec["id"] == "lgbm":
            for h in cfg.horizons:
                s = train_lgbm_horizon(cfg, h, universe_name, end_date)
                series_list.append(s)
        elif spec["id"] == "alstm":
            from production.train_alstm import train_alstm_multihead  # added in T13
            series_list.extend(train_alstm_multihead(cfg, universe_name, end_date))
        elif spec["id"] == "tra":
            from production.train_tra import train_tra_multihead  # added in T15
            series_list.extend(train_tra_multihead(cfg, universe_name, end_date))

    base_preds = pd.concat(series_list, axis=1).dropna(how="all")

    # Ensemble step — Phase E: Ridge stacking with OOF training, plus a
    # 3-level fallback chain: Ridge -> rank_average -> roll back to last week.
    from production.ensemble_stacker import RidgeStacker
    from production.ensemble_rank_avg import rank_average

    # OOF training data: pull realized 5d open-to-open labels for the valid
    # window. The beta phase approximates by reusing the valid window predictions
    # because all three base models early-stop on valid (not fit on it). A
    # dedicated Stack-fit window is a gamma improvement.
    try:
        from qlib.data import D
        h5 = next(h for h in cfg.horizons if h.name == "5d")
        s_5 = split(end_date=end_date, cfg=h5)
        label_expr = "Ref($open, -6) / Ref($open, -1) - 1"
        labels = D.features(
            instruments=universe_name,
            fields=[label_expr],
            start_time=str(s_5.valid_start),
            end_time=str(s_5.valid_end),
        )
        labels.columns = ["y"]
        labels.index.names = ["instrument", "datetime"]
        labels = labels.swaplevel("instrument", "datetime").sort_index()

        # base_preds restricted to the valid window
        valid_mask = (
            (base_preds.index.get_level_values("datetime") >= pd.Timestamp(s_5.valid_start))
            & (base_preds.index.get_level_values("datetime") <= pd.Timestamp(s_5.valid_end))
        )
        valid_base = base_preds[valid_mask]
        stacker = RidgeStacker().fit_oof(valid_base, labels["y"])
        unified = stacker.predict_with_fallback(base_preds).rename("score")
        _log.info("stacker_fitted_ok alpha=%s", stacker.alpha)
    except Exception as exc:
        _log.warning("stacker_failed_using_rank_average error=%s", str(exc))
        rank_avg_series = rank_average(base_preds)
        unified = (-rank_avg_series).rename("score")

    out = base_preds.copy()
    out["score"] = unified
    out["consensus"] = consensus_per_row(base_preds)
    out = ewma_smooth(out, alpha=cfg.ewma_alpha, score_col="score")

    pred_path = REPO_ROOT / "examples" / "mlruns" / f"pred_{end_date}.pkl"
    write_pred_pkl(out, pred_path)
    _log.info("pred_pkl_written path=%s rows=%d", str(pred_path), len(out))

    # Step ⑦ — Scorecard on the test window (if labels are available)
    try:
        from qlib.data import D
        h5 = next(h for h in cfg.horizons if h.name == "5d")
        s_5 = split(end_date=end_date, cfg=h5)
        labels = D.features(
            instruments=universe_name,
            fields=["Ref($open, -6) / Ref($open, -1) - 1"],
            start_time=str(s_5.test_start),
            end_time=str(s_5.test_end),
        )
        labels.columns = ["y"]
        labels.index.names = ["instrument", "datetime"]
        labels = labels.swaplevel().sort_index()["y"]

        from production.metrics import compute_scorecard
        score_window = out.reset_index().set_index(["datetime", "instrument"])["score"]
        scorecard = compute_scorecard(score_window, labels, top_k=30, bps=cfg.cost_bps)
        _log.info("scorecard %s", scorecard)
        from qlib.workflow import R
        with R.start(experiment_name=cfg.experiment_name, recorder_name=f"ensemble_{end_date}"):
            R.log_metrics(**{k: v for k, v in scorecard.items() if pd.notna(v)})
    except Exception as exc:
        _log.warning("scorecard_failed error=%s", str(exc))

    # Auto-archive recorders older than cfg.archive_weeks
    from production.mlruns_archive import archive_old_recorders
    mlruns_root = REPO_ROOT / "examples" / "mlruns"
    archive_root = REPO_ROOT / "production" / "archive"
    archived = archive_old_recorders(mlruns_root, archive_root, keep_weeks=cfg.archive_weeks)
    _log.info("recorders_archived count=%d", archived)

    # Auto-rollback: if past 2 weeks cumulative IR < 0, revert to N-2 recorder
    try:
        from qlib.workflow import R
        recs = sorted(
            R.list_recorders(experiment_name=cfg.experiment_name).values(),
            key=lambda r: r.info.get("start_time", ""),
            reverse=True,
        )
        if len(recs) >= 3:
            recent_irs = []
            for rr in recs[:2]:
                m = rr.list_metrics() if hasattr(rr, "list_metrics") else {}
                if "ir" in m:
                    recent_irs.append(m["ir"])
            if recent_irs and sum(recent_irs) < 0:
                _log.warning(
                    "auto_rollback_triggered current=%s rollback_to=%s recent_irs=%s",
                    recs[0].id, recs[2].id, recent_irs,
                )
                bad_rec_dir = mlruns_root / "1" / recs[0].id
                if bad_rec_dir.exists():
                    import shutil as _sh
                    _sh.move(str(bad_rec_dir), str(archive_root / "1" / f"rolled_back_{recs[0].id}"))
    except Exception as exc:
        _log.warning("rollback_check_failed error=%s", str(exc))

    return pred_path


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run-once")
    p_run.add_argument("--end-date", default=None)
    p_run.add_argument("--config", default="production/configs/rolling_ensemble.yaml")
    args = parser.parse_args()

    if args.cmd == "run-once":
        end = date.fromisoformat(args.end_date) if args.end_date else date.today()
        cfg = load_config(REPO_ROOT / args.config)
        path = run_once(cfg, end)
        print(f"OK: wrote {path}")
    else:
        raise NotImplementedError(args.cmd)


if __name__ == "__main__":
    main()
