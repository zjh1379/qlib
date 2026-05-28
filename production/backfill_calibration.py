"""One-off backfill: reload validation predictions + labels from latest
weekly recorders, fit isotonic calibration, save to
production/cache/latest_calibration.pkl.

Run once after deploying the calibration module so daily_inference and the
backend have a populated calibration file. Idempotent — overwrites
latest_calibration.pkl + writes a dated backup.

For pre-Task-3 recorders (no valid_pred.pkl/valid_label.pkl artifacts),
falls back to pred.pkl + computes labels from qlib handler. The resulting
calibration is slightly biased (test-set based), but acceptable as initial
backfill since it's only applied to FUTURE inferences on FUTURE dates.

Usage:
  python -m production.backfill_calibration
  python -m production.backfill_calibration --end-date 2026-05-22
"""
from __future__ import annotations

import sys as _sys
import sysconfig as _sysconfig
_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import argparse
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(REPO_ROOT))

from production.calibration import fit_calibration, save_calibration

log = logging.getLogger("backfill_calibration")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CACHE_DIR = REPO_ROOT / "production" / "cache"
LATEST_PATH = CACHE_DIR / "latest_calibration.pkl"


def _fit_and_save(pred_df: pd.DataFrame, label_df: pd.DataFrame,
                  out_path: Path, trained_at: str) -> dict:
    cal = fit_calibration(pred_df, label_df)
    save_calibration(cal, out_path, meta={
        "trained_at": trained_at,
        "saved_at": datetime.utcnow().isoformat(),
        "n_rows": int(len(pred_df)),
    })
    log.info("saved calibration to %s horizons=%s",
             out_path, list(cal.keys()))
    return cal


def _recorder_name(rec) -> str:
    info = rec.info if hasattr(rec, "info") else {}
    name = info.get("name") if isinstance(info, dict) else getattr(rec, "name", "")
    if name:
        return name
    try:
        return rec.client.get_run(rec.id).data.tags.get("mlflow.runName", "")
    except Exception:
        return ""


def _load_calibration_data_from_recorders(end_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each (model, horizon) load the matching <model>_<horizon>_<end_date>
    recorder. Try valid_pred.pkl/valid_label.pkl first (new artifacts from
    Task 3); fall back to pred.pkl + compute labels via qlib expression.
    """
    from qlib.workflow import R
    from production.rolling_train import load_config, init_qlib

    cfg = load_config(REPO_ROOT / "production/configs/rolling_ensemble.yaml")
    init_qlib(cfg)

    end_str = end_date.isoformat()
    exp = R.get_exp(experiment_name=cfg.experiment_name)
    recs = exp.list_recorders()
    if isinstance(recs, dict):
        recs = list(recs.values())

    pred_cols: dict[str, pd.Series] = {}
    label_cols: dict[str, pd.Series] = {}

    for model_id in ("lgbm", "alstm", "tra"):
        for h in ("1d", "5d", "20d"):
            target = f"{model_id}_{h}_{end_str}"
            matched = [r for r in recs if _recorder_name(r) == target]
            if not matched:
                log.warning("recorder_missing %s", target)
                continue
            rec = matched[0]
            try:
                try:
                    pred = rec.load_object("valid_pred.pkl")
                    use_valid = True
                except Exception:
                    # Pre-Task-3 recorders save the per-horizon test prediction
                    # as pred_<horizon>.pkl (recorder is already horizon-scoped).
                    pred = rec.load_object(f"pred_{h}.pkl")
                    use_valid = False
                    log.warning("recorder_lacks_valid_pred falling back to pred_%s.pkl %s",
                                h, target)
            except Exception as exc:
                log.warning("load_failed %s: %s", target, exc)
                continue

            if isinstance(pred, pd.DataFrame):
                pred = pred["score"] if "score" in pred.columns else pred.iloc[:, 0]
            col = f"{model_id}_{h}"
            pred_cols[col] = pred.rename(col)

            if f"label_{h}" not in label_cols:
                # Try saved label artifact first
                lab = None
                for fname in ("valid_label.pkl", "label.pkl"):
                    try:
                        lab = rec.load_object(fname)
                        break
                    except Exception:
                        continue
                if lab is None and not use_valid:
                    # Fallback: compute label from qlib data over the same date range
                    try:
                        from qlib.data.dataset.loader import QlibDataLoader
                        n = {"1d": 1, "5d": 5, "20d": 20}[h]
                        dates = pred.index.get_level_values("datetime")
                        instruments = pred.index.get_level_values("instrument").unique().tolist()
                        loader = QlibDataLoader(
                            config={"label": [[f"Ref($open, -{n}+1)/Ref($open, -1) - 1"],
                                              [f"label_{h}"]]},
                        )
                        lab = loader.load(instruments=instruments,
                                          start_time=dates.min(),
                                          end_time=dates.max())
                        log.info("label_computed_from_qlib for %s", target)
                    except Exception as exc:
                        log.warning("label_compute_failed for %s: %s", target, exc)

                if lab is not None:
                    if isinstance(lab, pd.DataFrame):
                        lab = lab.iloc[:, 0]
                    label_cols[f"label_{h}"] = lab.rename(f"label_{h}")

    pred_df = pd.concat(list(pred_cols.values()), axis=1).sort_index() \
              if pred_cols else pd.DataFrame()
    label_df = pd.concat(list(label_cols.values()), axis=1).sort_index() \
               if label_cols else pd.DataFrame()
    return pred_df, label_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD; default = 2026-05-22")
    args = parser.parse_args()

    end = date.fromisoformat(args.end_date) if args.end_date else date(2026, 5, 22)

    pred_df, label_df = _load_calibration_data_from_recorders(end)
    if pred_df.empty:
        log.error("no_pred_data — aborting")
        return 1
    if label_df.empty:
        log.error("no_label_data — aborting")
        return 1

    log.info("loaded pred shape=%s label shape=%s", pred_df.shape, label_df.shape)

    _fit_and_save(pred_df, label_df, LATEST_PATH, trained_at=end.isoformat())

    # Dated backup
    backup = CACHE_DIR / f"calibration_{end.isoformat()}.pkl"
    _fit_and_save(pred_df, label_df, backup, trained_at=end.isoformat())
    return 0


if __name__ == "__main__":
    sys.exit(main())
