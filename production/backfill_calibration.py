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


def _list_weekly_end_dates(recs, max_weeks: int = 12) -> list[str]:
    """Discover the distinct weekly end_dates (YYYY-MM-DD) we have recorders
    for, sorted descending. Looks for names matching <model>_<horizon>_<date>.

    Returns up to max_weeks most recent end_dates.
    """
    import re as _re
    pat = _re.compile(r"^(?:lgbm|alstm|tra)_(?:1d|5d|20d)_(\d{4}-\d{2}-\d{2})$")
    seen: set[str] = set()
    for r in recs:
        m = pat.match(_recorder_name(r))
        if m:
            seen.add(m.group(1))
    return sorted(seen, reverse=True)[:max_weeks]


def _load_calibration_data_from_recorders(
    end_date: date, *, weeks_back: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-recorder backfill: concatenate test slices from the last N weekly
    recorders, then compute forward labels FROM QLIB (qlib has the realized
    future prices that the old test slices were predicting).

    This solves the 'only 1716 samples for 5d, 0 for 20d' problem from the
    single-recorder approach: old test slices' T+5 / T+20 future prices are
    now in qlib's data range.

    Returns:
      pred_df: index (datetime, instrument), columns = {lgbm/alstm/tra}_{1d/5d/20d}
      label_df: index (datetime, instrument), columns = label_1d / label_5d / label_20d
    """
    from qlib.data.dataset.loader import QlibDataLoader
    from qlib.workflow import R
    from production.rolling_train import load_config, init_qlib

    cfg = load_config(REPO_ROOT / "production/configs/rolling_ensemble.yaml")
    init_qlib(cfg)

    exp = R.get_exp(experiment_name=cfg.experiment_name)
    recs = exp.list_recorders()
    if isinstance(recs, dict):
        recs = list(recs.values())

    # Discover the last N weekly end_dates (5/22, 5/15, 5/08, 5/01, 4/24, ...)
    weeks = _list_weekly_end_dates(recs, max_weeks=weeks_back)
    if end_date.isoformat() not in weeks:
        weeks = [end_date.isoformat()] + [w for w in weeks if w < end_date.isoformat()]
        weeks = weeks[:weeks_back]
    log.info("multi_recorder_backfill weeks=%d range=%s..%s",
             len(weeks), weeks[-1] if weeks else "?", weeks[0] if weeks else "?")

    # Per-model-per-horizon: concatenate pred_<h>.pkl from all matching weeks
    pred_by_col: dict[str, list[pd.Series]] = {
        f"{m}_{h}": [] for m in ("lgbm", "alstm", "tra") for h in ("1d", "5d", "20d")
    }
    all_instruments: set[str] = set()
    overall_min_date = None
    overall_max_date = None

    for wk in weeks:
        for model_id in ("lgbm", "alstm", "tra"):
            for h in ("1d", "5d", "20d"):
                target = f"{model_id}_{h}_{wk}"
                matched = [r for r in recs if _recorder_name(r) == target]
                if not matched:
                    continue
                rec = matched[0]
                try:
                    # Prefer valid_pred.pkl (post-Task-3 recorders), else test
                    try:
                        pred = rec.load_object("valid_pred.pkl")
                    except Exception:
                        pred = rec.load_object(f"pred_{h}.pkl")
                except Exception as exc:
                    log.warning("load_failed %s: %s", target, exc)
                    continue
                if isinstance(pred, pd.DataFrame):
                    pred = pred["score"] if "score" in pred.columns else pred.iloc[:, 0]
                pred_by_col[f"{model_id}_{h}"].append(pred)
                idx_dates = pred.index.get_level_values("datetime")
                idx_instr = pred.index.get_level_values("instrument").unique()
                all_instruments.update(idx_instr.tolist())
                dmin, dmax = idx_dates.min(), idx_dates.max()
                if overall_min_date is None or dmin < overall_min_date:
                    overall_min_date = dmin
                if overall_max_date is None or dmax > overall_max_date:
                    overall_max_date = dmax

    # Concatenate each col across weeks (de-dup overlapping dates by 'last')
    pred_cols: dict[str, pd.Series] = {}
    for col, parts in pred_by_col.items():
        if not parts:
            continue
        s = pd.concat(parts).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        pred_cols[col] = s.rename(col)
    if not pred_cols:
        return pd.DataFrame(), pd.DataFrame()

    pred_df = pd.concat(list(pred_cols.values()), axis=1).sort_index()

    # Compute forward labels FROM QLIB for the full date range × instruments.
    # qlib's expression engine handles NaN for out-of-range future prices
    # automatically — we just dropna later.
    label_cols: dict[str, pd.Series] = {}
    instruments_list = sorted(all_instruments)
    log.info(
        "computing labels via qlib date_range=%s..%s instruments=%d",
        overall_min_date, overall_max_date, len(instruments_list),
    )
    for h, n in (("1d", 1), ("5d", 5), ("20d", 20)):
        try:
            # MUST match training-time label formula in
            # production/custom_handler.py line 65/77:
            #   Ref($open, -(N+1)) / Ref($open, -1) - 1
            # = open at T+N+1 / open at T+1 - 1
            # = N-day forward return entered next-day open.
            #
            # The previous formula "Ref($open, -N+1) / Ref($open, -1)" was
            # wrong: at N=1 it evaluated to open(T)/open(T+1)-1 which is the
            # INVERSE direction, producing nonsensical isotonic outputs
            # like 1d pred_return = -26.75% for all top picks.
            label_expr = f"Ref($open, -{n + 1}) / Ref($open, -1) - 1"
            loader = QlibDataLoader(
                config={"label": (
                    [label_expr],
                    [f"label_{h}"],
                )},
            )
            lab = loader.load(
                instruments=instruments_list,
                start_time=str(overall_min_date.date()),
                end_time=str(overall_max_date.date()),
            )
            if isinstance(lab, pd.DataFrame):
                lab = lab.iloc[:, 0]
            label_cols[f"label_{h}"] = lab.rename(f"label_{h}")
            valid = lab.notna().sum()
            log.info("label_qlib horizon=%s non_nan=%d total=%d", h, valid, len(lab))
        except Exception as exc:
            log.warning("label_compute_failed horizon=%s: %s", h, exc)

    label_df = pd.concat(list(label_cols.values()), axis=1).sort_index() \
               if label_cols else pd.DataFrame()

    # Normalize index name conventions (qlib often returns instrument-then-datetime)
    if "datetime" in label_df.index.names and "instrument" in label_df.index.names:
        if label_df.index.names[0] == "instrument":
            label_df = label_df.swaplevel().sort_index()

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
