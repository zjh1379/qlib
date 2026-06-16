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

# --- IMPORTANT: sys.path fixup must run BEFORE any qlib import. ---
# This repo is a checkout of microsoft/qlib with an uncompiled qlib/ source
# directory at the worktree root. Running `python -m production.rolling_train`
# from that root puts the empty string (cwd) at sys.path[0], so `import qlib`
# finds the uncompiled source and fails on `qlib.data._libs.rolling`. We
# insert the conda-env site-packages at the front of sys.path so the
# installed qlib (with compiled .pyd extensions) wins.
import sys as _sys
import sysconfig as _sysconfig

_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import argparse
import logging
import sys
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from production.consensus import consensus_per_row, write_pred_pkl
from production.pit_constituents import load_or_refresh, members_on
from production.post_process import ewma_smooth
from production.progress import emit_progress, emit_recorder
from production.walk_forward import HorizonConfig, split

_log = logging.getLogger("rolling_train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = ("lgbm", "alstm", "tra")


def progress_total(cfg: "RollingConfig") -> int:
    """Total progress units for one run_once: one per enabled base model, plus
    three fixed stages (universe build, ensemble, done). Must stay in sync with
    the emit_progress calls in run_once."""
    enabled = [s for s in cfg.model_specs if s.get("enabled")]
    return len(enabled) + 3


def resolve_feature_handler(features: str) -> tuple[str, str]:
    """Map a --features choice to (handler_class_name, module_path).

    Returns:
        (handler_class_name, module_path) — suitable for use in handler_cfg
        and for selecting the handler class to instantiate.

    Raises:
        ValueError: if `features` is not a recognised choice.
    """
    if features == "alpha158":
        return ("Alpha158_OpenH", "custom_handler")
    if features == "shortterm":
        return ("AlphaShortTerm", "custom_handler")
    raise ValueError(f"unknown features {features!r} — expected 'alpha158' or 'shortterm'")


def backfill_fold_end_dates(start: date, end: date, step_weeks: int = 1) -> list[date]:
    """Friday-anchored fold end-dates from start..end, stepping step_weeks each time.

    Finds the first Friday on or after `start`, then yields dates at
    `step_weeks * 7` day intervals until `end` is exceeded.  Pure function
    (no I/O) — safe to unit-test without qlib.
    """
    days_to_friday = (4 - start.weekday()) % 7
    cursor = start + timedelta(days=days_to_friday)
    out: list[date] = []
    while cursor <= end:
        out.append(cursor)
        cursor += timedelta(days=step_weeks * 7)
    return out


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
    # Extra markets (e.g. 'etfs', 'custom') to union into the PIT csi800
    # universe — lets the trained model also produce predictions for these
    # symbols even though they're not in the csi300+csi500 base.
    universe_extras: list[str] | None = None


def _stack_score(base_preds: pd.DataFrame, oof_labels: pd.Series, *, ewma_alpha: float = 0.5) -> pd.DataFrame:
    """Fit the Ridge stacker on OOF (base_preds rows overlapping oof_labels dates)
    and score the full base_preds. Pure: no qlib/IO. Mirrors run_once's ensemble math."""
    from production.ensemble_stacker import RidgeStacker
    from production.ensemble_rank_avg import rank_average
    try:
        valid_mask = base_preds.index.get_level_values("datetime").isin(
            oof_labels.index.get_level_values("datetime").unique()
        )
        valid_base = base_preds[valid_mask]
        stacker = RidgeStacker().fit_oof(valid_base, oof_labels)
        unified = stacker.predict_with_fallback(base_preds).rename("score")
    except Exception as exc:
        _log.warning("stacker_failed_using_rank_average error=%s", str(exc))
        unified = (-rank_average(base_preds)).rename("score")
    out = base_preds.copy()
    out["score"] = unified
    out["consensus"] = consensus_per_row(base_preds)
    out = ewma_smooth(out, alpha=ewma_alpha, score_col="score")
    return out


def pool_stack_write(
    cfg: "RollingConfig", series_list: list[pd.Series], end_date: date, members: list[str],
    *, experiment_name: str, recorder_name: str, seed_serving: bool,
) -> Path:
    """Concat base series, refit stacker on the 5d valid window, write pred.pkl,
    and persist a recorder under experiment_name. Used by reblend_single (candidate exp)."""
    from qlib.data import D
    from qlib.workflow import R
    base_preds = pd.concat(series_list, axis=1).dropna(how="all")
    h5 = next(h for h in cfg.horizons if h.name == "5d")
    s_5 = split(end_date=end_date, cfg=h5)
    labels = D.features(instruments=members, fields=["Ref($open, -6) / Ref($open, -1) - 1"],
                        start_time=str(s_5.valid_start), end_time=str(s_5.valid_end))
    labels.columns = ["y"]
    labels.index.names = ["instrument", "datetime"]
    labels = labels.swaplevel("instrument", "datetime").sort_index()["y"]
    out = _stack_score(base_preds, labels, ewma_alpha=cfg.ewma_alpha)
    pred_path = REPO_ROOT / "examples" / "mlruns" / f"pred_{recorder_name}.pkl"
    write_pred_pkl(out, pred_path)
    with R.start(experiment_name=experiment_name, recorder_name=recorder_name):
        try:
            emit_recorder(R.get_recorder().id)
        except Exception:
            pass
        if seed_serving:
            R.save_objects(**{"pred.pkl": out})
    return pred_path


def load_config(path: Path) -> RollingConfig:
    # Explicit UTF-8 — Windows locale (GBK) chokes on non-ASCII comments
    # (e.g. arrow/multiplication-sign in smoke configs).
    with path.open(encoding="utf-8") as f:
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
        universe_extras=raw.get("universe_extras") or None,
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
        extra_markets=cfg.universe_extras,
    )
    # Add the extras to `members` too so the consensus/scorecard and
    # downstream label fetches include them.
    if cfg.universe_extras:
        from production.pit_constituents import _read_market_file
        extras: set[str] = set()
        for market in cfg.universe_extras:
            extras.update(_read_market_file(qlib_data_root, market))
        members = sorted(set(members) | extras)
    return members, universe_name


def train_lgbm_horizon(
    cfg: RollingConfig,
    horizon: HorizonConfig,
    universe_name: str,
    end_date: date,
    *,
    features: str = "alpha158",
    objective: str = "mse",
) -> pd.Series:
    """Train one LightGBM head and return its predictions on the test window.

    `universe_name` is the qlib instruments-file name produced by
    `build_universe()` — passed to the handler as a string so qlib resolves it
    via `D.instruments(market=universe_name)`.

    `features` selects the feature handler: "alpha158" (default, backward-
    compatible) uses Alpha158_OpenH; "shortterm" uses AlphaShortTerm which
    appends 6 non-redundant short-term factors on top of Alpha158.

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

    from custom_handler import Alpha158_OpenH, AlphaShortTerm  # noqa: E402

    _handler_cls_map = {
        "Alpha158_OpenH": Alpha158_OpenH,
        "AlphaShortTerm": AlphaShortTerm,
    }
    cls_name, module_path = resolve_feature_handler(features)
    handler = _handler_cls_map[cls_name](
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

    _lgbm_kwargs = dict(lgbm_yaml["model"]["kwargs"])
    if objective == "lambdarank":
        from production.lgbm_rank import LGBRankModel
        model = LGBRankModel(**_lgbm_kwargs)  # ignores `loss`, fixes objective=lambdarank
    else:
        model = LGBModel(**_lgbm_kwargs)
    # Handler config for daily_inference to rebuild features on new dates.
    # Strip segment-specific kwargs so daily_inference can substitute its
    # own start_time / end_time / instruments.
    handler_cfg = {
        "class": cls_name,
        "module_path": module_path,
        "kwargs": {
            "horizon_days": cfg.horizon_days[horizon.name],
            "fit_start_time": str(s.train_start),
            "fit_end_time": str(s.train_end),
            "instruments": universe_name,
        },
    }
    with R.start(experiment_name=cfg.experiment_name, recorder_name=f"lgbm_{horizon.name}_{end_date}"):
        model.fit(dataset)
        pred = model.predict(dataset)
        R.save_objects(**{f"pred_{horizon.name}.pkl": pred})
        # NEW: save calibration + inference artifacts (T3)
        from production.train_helpers import save_calibration_artifacts
        save_calibration_artifacts(model, dataset, handler_cfg)
    pred = pred.rename(f"lgbm_{horizon.name}")
    return pred


# ---------------------------------------------------------------------------
# T2 helpers — resumable backfill skip predicate
# ---------------------------------------------------------------------------

def _rec_name(r) -> str:
    """Extract the recorder name from a qlib recorder object (or a fake in tests)."""
    info = getattr(r, "info", {})
    if isinstance(info, dict) and info.get("name"):
        return info["name"]
    return getattr(r, "name", "") or ""


def fold_recorders_complete(exp, end_date: date, models, horizons) -> bool:
    """Return True if every <model>_<horizon>_<end_date> recorder already exists.

    Works with live qlib experiment objects and with the lightweight fake used
    in unit tests (any object with a `.list_recorders()` method returning an
    iterable of objects that have `.info["name"]` or `.name`).
    """
    recs = exp.list_recorders()
    # qlib returns a dict {id: recorder}; tests return a list — handle both.
    if isinstance(recs, dict):
        recs = recs.values()
    have = {_rec_name(r) for r in recs}
    es = end_date.isoformat()
    needed = {f"{m}_{h}_{es}" for m in models for h in horizons}
    return needed.issubset(have)


def run_once(
    cfg: RollingConfig,
    end_date: date,
    skip_pool: bool = False,
    *,
    test_weeks_override: int | None = None,
    train_years_override: int | None = None,
    skip_if_exists: bool = False,
    features: str = "alpha158",
    objective: str = "mse",
) -> Path | None:
    """Run one weekly iteration. Returns the path to the written pred.pkl,
    or None when `skip_pool=True` (caller is responsible for pooling).

    `skip_pool` is used by `production/run_split.py` to split a memory-heavy
    end-to-end run into per-model subprocesses: each subprocess trains its
    models (writes to mlflow recorders) and exits, releasing all memory.

    New keyword-only params (all optional, backward-compatible):
      test_weeks_override:  override HorizonConfig.test_weeks for all horizons
                            (e.g. 26 for semi-annual OOF windows).
      train_years_override: override HorizonConfig.train_years for all horizons
                            (e.g. 3 to limit lookback when data only starts 2018).
      skip_if_exists:       if True and all recorders for this fold already exist
                            in the experiment, skip training and return the
                            existing pred pickle (if present) or fall through to
                            retrain if the pickle is missing.
      features:             feature handler selection: "alpha158" (default,
                            backward-compatible) or "shortterm" (Alpha158 + 6
                            non-redundant short-term factors).
    """
    # Apply horizon overrides (creates a shadow cfg — does not mutate the original).
    if test_weeks_override is not None or train_years_override is not None:
        new_horizons = [
            replace(
                h,
                test_weeks=test_weeks_override if test_weeks_override is not None else h.test_weeks,
                train_years=train_years_override if train_years_override is not None else h.train_years,
            )
            for h in cfg.horizons
        ]
        cfg = replace(cfg, horizons=new_horizons)

    # Resumable backfill: skip fold when all recorders already exist.
    if skip_if_exists:
        try:
            from qlib.workflow import R
            init_qlib(cfg)
            exp = R.get_exp(experiment_name=cfg.experiment_name)
            enabled_models = tuple(
                s["id"] for s in cfg.model_specs if s["enabled"]
            )
            if fold_recorders_complete(
                exp, end_date,
                enabled_models,
                tuple(h.name for h in cfg.horizons),
            ):
                pred_path = REPO_ROOT / "examples" / "mlruns" / f"pred_{end_date}.pkl"
                if pred_path.exists():
                    _log.info("skip_existing_fold end_date=%s pred_path=%s", end_date, pred_path)
                    return pred_path
                _log.info(
                    "recorders_exist_but_pred_missing end_date=%s — retraining",
                    end_date,
                )
        except Exception as exc:
            _log.warning("skip_if_exists_check_failed end_date=%s error=%s — proceeding", end_date, exc)

    init_qlib(cfg)
    members, universe_name = build_universe(cfg, end_date)
    _log.info(
        "universe_built size=%d as_of=%s name=%s",
        len(members), str(end_date), universe_name,
    )

    total = progress_total(cfg)
    step = 1
    emit_progress("universe", step, total, f"universe {len(members)} stocks")

    # Train all enabled base models for all horizons
    series_list: list[pd.Series] = []
    for spec in cfg.model_specs:
        if not spec["enabled"]:
            continue
        step += 1
        emit_progress("train", step, total, f"training {spec['id']}")
        if spec["id"] == "lgbm":
            for h in cfg.horizons:
                s = train_lgbm_horizon(cfg, h, universe_name, end_date, features=features, objective=objective)
                series_list.append(s)
        elif spec["id"] == "alstm":
            from production.train_alstm import train_alstm_multihead  # added in T13
            series_list.extend(train_alstm_multihead(cfg, universe_name, end_date))
        elif spec["id"] == "tra":
            from production.train_tra import train_tra_multihead  # added in T15
            series_list.extend(train_tra_multihead(cfg, universe_name, end_date))

    # Split-run support: when run_split.py launches us per-model, return early
    # before the ensemble step. Each model's predictions are already saved to
    # its mlflow recorder; the orchestrator pools them after all subprocesses
    # finish.
    if skip_pool:
        _log.info("skip_pool=True, %d series trained — returning early", len(series_list))
        return None

    base_preds = pd.concat(series_list, axis=1).dropna(how="all")

    step += 1
    emit_progress("ensemble", step, total, "stacking ensemble")

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
        # NOTE: pass `members` (list of symbol strings) — not `universe_name`
        # (a qlib instruments-file name). The handler-based code path (LGBM
        # train) accepts the string because the handler wraps it via
        # D.instruments(market=...), but raw D.features needs a list or a
        # dict from D.instruments(). The list is the simpler choice.
        labels = D.features(
            instruments=members,
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

    # Step ⑦ — Scorecard on the test window (best-effort; needs FUTURE labels,
    # which don't exist yet for the most recent live dates -> scorecard stays {}).
    scorecard: dict = {}
    try:
        from qlib.data import D
        h5 = next(h for h in cfg.horizons if h.name == "5d")
        s_5 = split(end_date=end_date, cfg=h5)
        # Same `members` vs `universe_name` rationale as the stacker block above.
        labels = D.features(
            instruments=members,
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
    except Exception as exc:
        _log.warning("scorecard_failed error=%s", str(exc))

    # Step ⑦b — Persist the ensemble_<end_date> recorder ONCE. For LIVE folds,
    # seed pred.pkl so the backend (get_latest_recorder_id) + daily_inference can
    # serve/extend it — the missing link that froze live serving. Historical
    # backfill folds get metrics only (seeding them would hijack serving via
    # start_time ordering with a stale data date).
    try:
        from qlib.workflow import R
        from production.train_helpers import is_live_fold
        with R.start(experiment_name=cfg.experiment_name, recorder_name=f"ensemble_{end_date}"):
            try:
                emit_recorder(R.get_recorder().id)
            except Exception:
                pass
            if is_live_fold(end_date):
                R.save_objects(**{"pred.pkl": out})
                _log.info("seeded_serving_pred recorder=ensemble_%s rows=%d", end_date, len(out))
            if scorecard:
                R.log_metrics(**{k: v for k, v in scorecard.items() if pd.notna(v)})
    except Exception as exc:
        _log.warning("ensemble_recorder_persist_failed error=%s", str(exc))

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

    emit_progress("done", total, total, "done")
    return pred_path


def run_backfill(
    cfg: RollingConfig,
    start: date,
    end: date,
    *,
    step_weeks: int = 1,
    test_weeks_override: int | None = None,
    train_years_override: int | None = None,
    skip_if_exists: bool = True,
    features: str = "alpha158",
    objective: str = "mse",
) -> list[Path]:
    """Loop run_once over every Friday in [start, end]. Writes one
    pred_<friday>.pkl per iteration. Returns list of written paths.

    Each iteration is independent — if one fails, we log and continue.

    New keyword-only params (all optional, backward-compatible):
      step_weeks:           advance the cursor by this many weeks per fold
                            (default 1 = weekly, unchanged from prior behavior).
      test_weeks_override:  forwarded to run_once (long test window support).
      train_years_override: forwarded to run_once (limits lookback; useful when
                            data only starts 2018 and we need 2021+ OOF).
      skip_if_exists:       forwarded to run_once; default True so backfill
                            runs are resumable by default.
      features:             forwarded to run_once; selects the LGBM feature
                            handler ("alpha158" default or "shortterm").
    """
    fold_dates = backfill_fold_end_dates(start, end, step_weeks)
    total = len(fold_dates)
    paths: list[Path] = []
    failures: list[tuple[date, str]] = []
    _log.info(
        "backfill_start start=%s end=%s step_weeks=%d total_folds=%d",
        start, end, step_weeks, total,
    )
    for iteration, cursor in enumerate(fold_dates, 1):
        _log.info("backfill_fold iteration=%d/%d end_date=%s", iteration, total, cursor)
        try:
            path = run_once(
                cfg, cursor,
                test_weeks_override=test_weeks_override,
                train_years_override=train_years_override,
                skip_if_exists=skip_if_exists,
                features=features,
                objective=objective,
            )
            if path is not None:
                paths.append(path)
            _log.info("backfill_fold_ok iteration=%d end_date=%s", iteration, cursor)
        except Exception as exc:
            _log.warning(
                "backfill_fold_failed iteration=%d end_date=%s error=%s",
                iteration, cursor, str(exc),
            )
            failures.append((cursor, str(exc)))
    _log.info("backfill_done folds_ok=%d folds_failed=%d", len(paths), len(failures))
    if failures:
        for d, e in failures:
            _log.warning("backfill_failure_summary date=%s error=%s", d, e[:200])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run-once")
    p_run.add_argument("--end-date", default=None)
    p_run.add_argument("--config", default="production/configs/rolling_ensemble.yaml")
    p_run.add_argument(
        "--only-models",
        default=None,
        help=(
            "Comma-separated subset of model ids (lgbm,alstm,tra) to train. "
            "Other models in cfg are disabled for this invocation. Lets us "
            "split a heavy training run across multiple subprocesses so each "
            "model starts with a clean memory state (run_split.py)."
        ),
    )
    p_run.add_argument(
        "--skip-pool",
        action="store_true",
        help=(
            "Skip the ensemble pool + pred_<date>.pkl write step. Use when "
            "splitting a run across multiple subprocesses — the orchestrator "
            "(run_split.py) does the pool after all models are trained, by "
            "reading the per-model mlflow recorders."
        ),
    )
    p_run.add_argument(
        "--features",
        choices=["alpha158", "shortterm"],
        default="shortterm",
        help=(
            "Feature handler for the LGBM model. 'shortterm' (DEFAULT since "
            "2026-06-03) = AlphaShortTerm (Alpha158 + 6 short-term factors; "
            "+12pp net CAGR / Calmar 0.34->0.77 vs alpha158, see "
            "docs/.../2026-06-03-shortterm-factors-results.md). run_split passes "
            "no --features, so this default is what production retrains use. "
            "'alpha158' selects the original Alpha158_OpenH (no short-term factors)."
        ),
    )

    p_run.add_argument(
        "--objective",
        choices=["mse", "lambdarank"],
        default="mse",
        help=(
            "LGBM objective: 'mse' (default, regression) or 'lambdarank' "
            "(learning-to-rank; optimizes top-of-ranking via production.lgbm_rank.LGBRankModel)."
        ),
    )

    p_back = sub.add_parser("backfill",
        help="Loop run-once over every Friday in [start, end]. ~5-30 min per week.")
    p_back.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p_back.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p_back.add_argument("--config", default="production/configs/rolling_ensemble.yaml")
    p_back.add_argument(
        "--step-weeks",
        type=int,
        default=1,
        help=(
            "Advance the fold cursor by this many weeks per iteration "
            "(default 1 = weekly). Use 26 for semi-annual OOF, 52 for annual."
        ),
    )
    p_back.add_argument(
        "--test-weeks",
        type=int,
        default=None,
        help=(
            "Override HorizonConfig.test_weeks for all horizons. "
            "Set equal to --step-weeks for non-overlapping OOF windows."
        ),
    )
    p_back.add_argument(
        "--train-years",
        type=int,
        default=None,
        help=(
            "Override HorizonConfig.train_years for all horizons. "
            "Use 3 when qlib data only starts 2018 and you want 2021+ eval."
        ),
    )
    p_back.add_argument(
        "--only-models",
        default=None,
        help=(
            "Comma-separated subset of model ids (lgbm,alstm,tra) to train. "
            "Other models in cfg are disabled for this invocation."
        ),
    )
    p_back.add_argument(
        "--features",
        choices=["alpha158", "shortterm"],
        default="shortterm",
        help=(
            "Feature handler for the LGBM model. 'shortterm' (DEFAULT since "
            "2026-06-03) = AlphaShortTerm (Alpha158 + 6 short-term factors; "
            "+12pp net CAGR / Calmar 0.34->0.77 vs alpha158, see "
            "docs/.../2026-06-03-shortterm-factors-results.md). run_split passes "
            "no --features, so this default is what production retrains use. "
            "'alpha158' selects the original Alpha158_OpenH (no short-term factors)."
        ),
    )

    p_back.add_argument(
        "--objective",
        choices=["mse", "lambdarank"],
        default="mse",
        help="LGBM objective: 'mse' (default) or 'lambdarank' (learning-to-rank).",
    )

    p_reblend = sub.add_parser("reblend", help="Retrain ONE model on the latest fold + re-blend into a candidate.")
    p_reblend.add_argument("--only", required=True, help="model id: lgbm | alstm | tra")
    p_reblend.add_argument("--end-date", default=None, help="fold end date; default = latest ensemble recorder's date")
    p_reblend.add_argument("--config", default="production/configs/rolling_ensemble.yaml")

    args = parser.parse_args()

    if args.cmd == "run-once":
        end = date.fromisoformat(args.end_date) if args.end_date else date.today()
        cfg = load_config(REPO_ROOT / args.config)
        # Filter cfg.model_specs to the requested subset (split-mode).
        if args.only_models:
            wanted = {m.strip() for m in args.only_models.split(",") if m.strip()}
            for spec in cfg.model_specs:
                if spec["id"] not in wanted:
                    spec["enabled"] = False
            _log.info("run_once_only_models=%s", sorted(wanted))
        path = run_once(cfg, end, skip_pool=args.skip_pool, features=args.features, objective=args.objective)
        if path is not None:
            print(f"OK: wrote {path}")
        else:
            print("OK: training done, pool skipped")
    elif args.cmd == "backfill":
        cfg = load_config(REPO_ROOT / args.config)
        # Filter cfg.model_specs to the requested subset (if --only-models given).
        if args.only_models:
            wanted = {m.strip() for m in args.only_models.split(",") if m.strip()}
            for spec in cfg.model_specs:
                if spec["id"] not in wanted:
                    spec["enabled"] = False
            _log.info("backfill_only_models=%s", sorted(wanted))
        paths = run_backfill(
            cfg,
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
            step_weeks=args.step_weeks,
            test_weeks_override=args.test_weeks,
            train_years_override=args.train_years,
            features=args.features,
            objective=args.objective,
        )
        print(f"OK: backfilled {len(paths)} folds")
        for p in paths:
            print(f"  {p}")
    elif args.cmd == "reblend":
        cfg = load_config(REPO_ROOT / args.config)
        from production.reblend import reblend_single, latest_full_fold_end_date
        end = date.fromisoformat(args.end_date) if args.end_date else latest_full_fold_end_date(cfg)
        name = reblend_single(cfg, args.only, end)
        print(f"OK: wrote candidate {name}")
    else:
        raise NotImplementedError(args.cmd)


if __name__ == "__main__":
    main()
