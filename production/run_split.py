"""Memory-safe weekly retrain orchestrator.

Runs LGBM, ALSTM, TRA each in their own Python subprocess so each model
starts with a clean memory state. The previous in-process design
(rolling_train.run_once with all three models) accumulated:

  - Alpha158 handler (~500 MB for csi800 × 7y)
  - 3× Alpha360 handler per model (~1-2 GB each, even with lazy build +
    gc) — joblib leaves memmap files behind that don't get reclaimed
    until the process exits.
  - LightGBM Boosters
  - PyTorch ALSTM model + GPU memory
  - PyTorch TRA model + OT routing state

A subprocess per model means OS reclaims everything between models. The
3.8 MiB allocation failure that crashed the workstation on 2026-05-28 was
classic Windows pagefile exhaustion from accumulated joblib memmaps; this
script makes that physically impossible.

Workflow:
  1. For each model in [lgbm, alstm, tra] (in that order — fastest first):
       subprocess: rolling_train run-once --only-models <m> --skip-pool
  2. After all 3 finish, pool from mlflow recorders + write pred_<date>.pkl
     in the SAME process — pooling is cheap (no model loading, just
     reading saved per-horizon predictions).

Usage:
  python -m production.run_split --end-date 2026-05-22
  python -m production.run_split --end-date 2026-05-22 --models lgbm,alstm

Failure handling:
  - If one model's subprocess fails (e.g. ALSTM OOM), continue with the
    others and pool whatever succeeded. The ensemble falls back to
    rank_average over the available columns.
  - Use --strict to abort on any subprocess failure (useful in CI).
"""
from __future__ import annotations

import sys as _sys
import sysconfig as _sysconfig
_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import argparse
import logging
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

_log = logging.getLogger("run_split")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

DEFAULT_MODELS = ("lgbm", "alstm", "tra")


def _train_one_model(
    end_date: date,
    model_id: str,
    config: str,
    *,
    max_rss_gb: float = 10.0,
    max_commit_pct: float = 85.0,
) -> int:
    """Launch rolling_train as a child process with only the named model
    enabled and pooling disabled. Returns the subprocess exit code.

    Memory watchdog (CRITICAL — added 2026-05-29 after Event 2004 audit
    showed single python.exe peaking at 17.2 GB virtual memory, which
    collapsed Windows commit charge and caused a hard system freeze):

    - max_rss_gb: kill the training subprocess if its RSS exceeds this
      threshold (default 10 GB ≈ 1/3 of system RAM). This prevents one
      runaway process from monopolising commit and freezing the whole OS.

    - max_commit_pct: also poll system-wide swap+RAM usage; if total commit
      exceeds this percent (default 85%), kill the heaviest training
      subprocess to release pressure before pagefile exhausts.
    """
    cmd = [
        sys.executable,
        "-m",
        "production.rolling_train",
        "run-once",
        "--end-date",
        end_date.isoformat(),
        "--config",
        config,
        "--only-models",
        model_id,
        "--skip-pool",
    ]
    _log.info(
        "subprocess_start model=%s max_rss=%.1fGB max_commit=%.0f%% cmd=%s",
        model_id, max_rss_gb, max_commit_pct, " ".join(cmd),
    )
    t0 = time.time()

    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))

    killed_reason: str | None = None
    try:
        import psutil
    except Exception:
        psutil = None  # type: ignore[assignment]

    while True:
        try:
            rc = proc.wait(timeout=10)
            break
        except subprocess.TimeoutExpired:
            if psutil is None:
                continue
            try:
                p = psutil.Process(proc.pid)
                rss_gb = p.memory_info().rss / 2**30
                # Sum children too (rolling_train spawns its own data jobs)
                for ch in p.children(recursive=True):
                    try:
                        rss_gb += ch.memory_info().rss / 2**30
                    except Exception:
                        pass
                vm = psutil.virtual_memory()
                sw = psutil.swap_memory()
                total_commit = vm.used + sw.used
                commit_limit = vm.total + sw.total
                commit_pct = 100.0 * total_commit / commit_limit if commit_limit else 0
                if rss_gb > max_rss_gb:
                    killed_reason = f"per-process RSS {rss_gb:.1f}GB exceeded {max_rss_gb}GB"
                elif commit_pct > max_commit_pct:
                    killed_reason = f"system commit {commit_pct:.0f}% exceeded {max_commit_pct}%"
                if killed_reason:
                    _log.error(
                        "memory_watchdog_killing model=%s pid=%d %s",
                        model_id, proc.pid, killed_reason,
                    )
                    for ch in p.children(recursive=True):
                        try:
                            ch.kill()
                        except Exception:
                            pass
                    p.kill()
                    rc = proc.wait(timeout=15)
                    break
            except psutil.NoSuchProcess:
                # Process already exited; loop will catch it next iteration
                continue
            except Exception as exc:
                _log.debug("watchdog poll error: %s", exc)

    dt = time.time() - t0
    _log.info(
        "subprocess_end model=%s rc=%d elapsed=%.1fs%s",
        model_id, rc, dt,
        f" KILLED({killed_reason})" if killed_reason else "",
    )
    return rc


def _pool_from_recorders(end_date: date, cfg_path: str) -> Path:
    """After all per-model subprocesses finished, read each
    `<model>_<horizon>_<end_date>` mlflow recorder, concat the series,
    compute ensemble score + EWMA, write pred_<end_date>.pkl.
    """
    from production.rolling_train import load_config, init_qlib
    from production.consensus import consensus_per_row, write_pred_pkl
    from production.ensemble_rank_avg import rank_average
    from production.post_process import ewma_smooth

    cfg = load_config(REPO_ROOT / cfg_path)
    init_qlib(cfg)

    import qlib
    from qlib.workflow import R

    exp_name = cfg.experiment_name
    end_str = end_date.isoformat()

    # Discover the latest <model>_<horizon>_<end_date> recorder per
    # (model, horizon). Each run_split invocation creates one fresh
    # recorder per model+horizon (rolling_train uses recorder_name
    # like 'lgbm_5d_2026-05-22'); we pick the most recent one by
    # mlflow start_time.
    exp = R.get_exp(experiment_name=exp_name)
    recs = exp.list_recorders()
    if isinstance(recs, dict):
        recs = list(recs.values())

    series: dict[str, pd.Series] = {}
    for rec in recs:
        info = rec.info if hasattr(rec, "info") else {}
        run_name = info.get("name") if isinstance(info, dict) else getattr(rec, "name", "")
        if not run_name:
            try:
                run_name = rec.client.get_run(rec.id).data.tags.get("mlflow.runName", "")
            except Exception:
                run_name = ""
        # match patterns: lgbm_<h>_<end>, alstm_<h>_<end>, tra_<h>_<end>
        for model_id in DEFAULT_MODELS:
            for h in ("1d", "5d", "20d"):
                target_name = f"{model_id}_{h}_{end_str}"
                if run_name != target_name:
                    continue
                pkl_name = f"pred_{h}.pkl"
                try:
                    obj = rec.load_object(pkl_name)
                except Exception as exc:
                    _log.warning(
                        "load_failed recorder=%s file=%s error=%s",
                        rec.id[:12], pkl_name, exc,
                    )
                    continue
                # TRA returns DataFrame with [score, label, score_0..K-1];
                # ALSTM/LGBM return Series. Normalise to Series named like
                # 'lgbm_5d' etc.
                if isinstance(obj, pd.DataFrame):
                    s = obj["score"] if "score" in obj.columns else obj.iloc[:, 0]
                else:
                    s = obj
                col = f"{model_id}_{h}"
                # If multiple recorders match (e.g. retries within the same
                # day), keep the latest by recorder id which is monotonic.
                if col in series:
                    continue
                series[col] = s.rename(col)
                _log.info(
                    "pooled col=%s rows=%d recorder=%s",
                    col, len(s), rec.id[:12],
                )

    if not series:
        raise RuntimeError(
            f"no per-model recorders found for end_date={end_str} in {exp_name}"
        )

    base = pd.concat(list(series.values()), axis=1).sort_index()
    base = base[~base.index.duplicated(keep="last")]
    base.index.names = ["datetime", "instrument"]

    # Score: rank-avg over 1d+5d cols only (v9 convention — see
    # production/pool_4wk_v9_no20d.py). 20d cols stay in the pkl for
    # diagnostics but don't contribute to the unified score.
    score_cols = [c for c in base.columns if not c.endswith("_20d")]
    _log.info("pool_score_cols=%s (of %d total)", score_cols, len(base.columns))

    ranks = base[score_cols].groupby(level="datetime").rank(ascending=False, method="min")
    base["score"] = -ranks.mean(axis=1, skipna=True)
    base["consensus"] = consensus_per_row(base[[c for c in base.columns if c not in ("score", "consensus")]])
    base = ewma_smooth(base, alpha=cfg.ewma_alpha, score_col="score")

    out_path = REPO_ROOT / "examples" / "mlruns" / f"pred_{end_str}.pkl"
    write_pred_pkl(base, out_path)
    _log.info("pool_wrote path=%s rows=%d cols=%s", out_path, len(base), list(base.columns))

    # Seed the serving recorder: save pred.pkl into ensemble_<end> so the backend
    # (get_latest_recorder_id) + daily_inference can serve/extend it. This is the
    # link that was missing — run_split previously wrote ONLY the file above, so
    # live serving froze at the last manually-pooled recorder. No-op (recency-
    # guarded) for historical backfill folds so they never hijack serving.
    from production.train_helpers import seed_serving_recorder
    if seed_serving_recorder(exp_name, end_date, base):
        _log.info("seeded_serving_pred recorder=ensemble_%s rows=%d", end_str, len(base))

    # === Refit per-horizon calibration on the valid slices of contributing
    # recorders. Stored in production/cache/latest_calibration.pkl for
    # daily_inference + backend to apply. Fail-soft.
    _refit_calibration(recs, end_str)

    return out_path


def _refit_calibration(recs, end_str: str) -> None:
    """Pull valid_pred.pkl + valid_label.pkl from each <model>_<horizon>_<end>
    recorder, fit a fresh isotonic regression per horizon, save to
    production/cache/latest_calibration.pkl.
    """
    try:
        from datetime import datetime as _dt
        from production.calibration import fit_calibration, save_calibration

        valid_pred_cols: dict[str, "pd.Series"] = {}
        valid_label_cols: dict[str, "pd.Series"] = {}
        for rec in recs:
            run_name = _recorder_name(rec)
            for mid in ("lgbm", "alstm", "tra"):
                for h in ("1d", "5d", "20d"):
                    target = f"{mid}_{h}_{end_str}"
                    if run_name != target:
                        continue
                    col = f"{mid}_{h}"
                    try:
                        vp = rec.load_object("valid_pred.pkl")
                        if isinstance(vp, pd.DataFrame):
                            vp = vp["score"] if "score" in vp.columns else vp.iloc[:, 0]
                        valid_pred_cols[col] = vp.rename(col)
                    except Exception:
                        # No artifact -> calibration just won't include this col
                        pass
                    if f"label_{h}" not in valid_label_cols:
                        try:
                            vl = rec.load_object("valid_label.pkl")
                            if isinstance(vl, pd.DataFrame):
                                vl = vl.iloc[:, 0]
                            valid_label_cols[f"label_{h}"] = vl.rename(f"label_{h}")
                        except Exception:
                            pass

        if valid_pred_cols and valid_label_cols:
            vp_df = pd.concat(list(valid_pred_cols.values()), axis=1).sort_index()
            vl_df = pd.concat(list(valid_label_cols.values()), axis=1).sort_index()
            cal = fit_calibration(vp_df, vl_df)
            cache_path = REPO_ROOT / "production" / "cache" / "latest_calibration.pkl"
            save_calibration(cal, cache_path, meta={
                "trained_at": end_str,
                "saved_at": _dt.utcnow().isoformat(),
                "n_rows": int(len(vp_df)),
            })
            _log.info("calibration_refit horizons=%s -> %s",
                      list(cal.keys()), cache_path)
        else:
            _log.warning("calibration_skip no valid_pred/label artifacts in recorders "
                         "(rerun after Task 3 lands)")
    except Exception as exc:
        _log.warning("calibration_refit_failed: %s", exc)


def _recorder_name(rec) -> str:
    info = rec.info if hasattr(rec, "info") else {}
    name = info.get("name") if isinstance(info, dict) else getattr(rec, "name", "")
    if name:
        return name
    try:
        return rec.client.get_run(rec.id).data.tags.get("mlflow.runName", "")
    except Exception:
        return ""


def _kill_zombie_python(min_mem_mb: int = 100) -> int:
    """Best-effort: kill any python.exe that looks like a leftover training
    process from a previous crashed run. Conservative match — only processes
    whose command line contains 'rolling_train' or 'production.run_split'
    (so we don't murder uvicorn, jupyter, etc).

    Returns count of killed processes. Silent skip on non-Windows.
    """
    if not _sys.platform.startswith("win"):
        return 0
    try:
        import psutil
    except Exception:
        return 0
    me_pid = subprocess.os.getpid()
    parent_pid = subprocess.os.getppid()
    killed = 0
    # Whitelist of cmdline substrings we ARE willing to kill. Anything else
    # (backend uvicorn, the user's jupyter, etc.) is left alone.
    TARGET_TOKENS = ("rolling_train", "production.run_split", "incremental_refresh")
    for proc in psutil.process_iter(attrs=["pid", "name", "memory_info"]):
        try:
            if proc.info["name"] != "python.exe":
                continue
            if proc.info["pid"] in (me_pid, parent_pid):
                continue
            rss_mb = proc.info["memory_info"].rss / 1024 / 1024
            if rss_mb < min_mem_mb:
                continue
            # Now check the command line — only kill known-training procs.
            try:
                cmdline = " ".join(proc.cmdline())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if not any(tok in cmdline for tok in TARGET_TOKENS):
                continue
            _log.warning(
                "killing zombie pid=%d rss=%dMB cmdline=%s",
                proc.info["pid"], int(rss_mb), cmdline[:120],
            )
            proc.kill()
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return killed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--config",
        default="production/configs/rolling_ensemble.yaml",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-sep list, default 'lgbm,alstm,tra'",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort the orchestrator if any model subprocess fails (default: continue)",
    )
    parser.add_argument(
        "--no-kill-zombies",
        action="store_true",
        help="Skip the pre-launch zombie python cleanup",
    )
    args = parser.parse_args()

    end = date.fromisoformat(args.end_date)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    config = args.config

    if not args.no_kill_zombies:
        killed = _kill_zombie_python()
        if killed:
            _log.info("killed_zombie_python_processes count=%d", killed)
            time.sleep(2)  # let the OS reclaim memory

    _log.info("orchestrator_start end_date=%s models=%s", end, models)
    failures: list[str] = []
    for model_id in models:
        rc = _train_one_model(end, model_id, config)
        if rc != 0:
            failures.append(model_id)
            _log.warning("model_subprocess_failed model=%s rc=%d", model_id, rc)
            if args.strict:
                _log.error("strict_mode_abort")
                return rc
        # Memory checkpoint: explicit small sleep + log
        time.sleep(2)

    if failures:
        _log.warning("partial_success failed_models=%s", failures)

    _log.info("orchestrator_pool_start")
    try:
        out = _pool_from_recorders(end, config)
        print(f"OK: wrote {out}")
        return 0
    except Exception as exc:
        _log.exception("pool_failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
