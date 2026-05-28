"""Re-train ALSTM × 3 horizons for specified Fridays without touching LGBM / TRA.

Use case: ALSTM YAML changed (e.g. CSRankNorm reverted) but LGBM / TRA artifacts
from the previous backfill are still valid. Re-running the full rolling_train
takes ~2h/week; this script targets only ALSTM (~25 min/week × 4 weeks ~1.5h).

For each end_date:
  1. Build the universe + walk-forward split (same as rolling_train).
  2. Call train_alstm_multihead to fit ALSTM_1d/5d/20d and save the alstm_*.pkl
     under fresh `alstm_<horizon>_<end_date>` recorders.
  3. Read the existing pred_<end_date>.pkl (must already contain lgbm_* and
     tra_* columns from the prior backfill).
  4. Replace its alstm_* columns with the freshly trained predictions.
  5. Recompute score = -rank_avg(base_preds.mean), consensus = top-K fraction.
  6. Run EWMA smoothing.
  7. Write back to pred_<end_date>.pkl.

Usage:
  python -m production.retrain_alstm_only --dates 2026-05-01,2026-05-08,2026-05-22

Notes:
- Skips end_dates that have no existing pred file (those need a full run-once
  via rolling_train first).
- Preserves LGBM / TRA columns exactly as they were.
"""
from __future__ import annotations

# IMPORTANT: force conda-env qlib over in-repo qlib/ source.
import sys as _sys
import sysconfig as _sysconfig
_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_log = logging.getLogger("retrain_alstm")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dates",
        required=True,
        help="Comma-separated YYYY-MM-DD list, e.g. 2026-05-01,2026-05-08,2026-05-22",
    )
    parser.add_argument(
        "--config",
        default="production/configs/rolling_ensemble.yaml",
    )
    args = parser.parse_args()

    from production.rolling_train import load_config, init_qlib, build_universe
    from production.train_alstm import train_alstm_multihead
    from production.consensus import consensus_per_row, write_pred_pkl
    from production.ensemble_rank_avg import rank_average
    from production.post_process import ewma_smooth

    cfg = load_config(REPO / args.config)
    init_qlib(cfg)

    dates = [date.fromisoformat(d) for d in args.dates.split(",")]
    mlruns_dir = REPO / "examples" / "mlruns"

    for end_date in dates:
        _log.info("=== retraining ALSTM for end_date=%s ===", end_date)

        pred_path = mlruns_dir / f"pred_{end_date}.pkl"
        if not pred_path.exists():
            _log.warning("skip end_date=%s — no existing pred file (run full rolling_train first)", end_date)
            continue

        # Read existing predictions (we'll overwrite only the ALSTM cols)
        existing = pd.read_pickle(pred_path)
        lgbm_cols = [c for c in existing.columns if c.startswith("lgbm_")]
        tra_cols = [c for c in existing.columns if c.startswith("tra_")]
        if not lgbm_cols:
            _log.warning("skip end_date=%s — no lgbm_* columns in existing pred file", end_date)
            continue
        _log.info(
            "existing pred_%s: %s rows, kept cols %s",
            end_date, len(existing), lgbm_cols + tra_cols,
        )

        # Rebuild universe for this end_date
        members, universe_name = build_universe(cfg, end_date)
        _log.info("universe_built size=%d as_of=%s", len(members), end_date)

        # Train ALSTM × 3 horizons → returns list of 3 Series (alstm_1d, _5d, _20d)
        alstm_series = train_alstm_multihead(cfg, universe_name, end_date)
        if not alstm_series:
            _log.error("end_date=%s: ALSTM produced no series", end_date)
            continue
        alstm_df = pd.concat(alstm_series, axis=1)
        _log.info("alstm trained: shape=%s cols=%s", alstm_df.shape, list(alstm_df.columns))

        # Reindex ALSTM predictions to the existing index (drop rows not in original)
        # New ALSTM may have slightly different index from existing (different stocks
        # listed, etc) — only keep the intersection.
        common_idx = existing.index.intersection(alstm_df.index)
        if len(common_idx) < 0.8 * len(existing):
            _log.warning(
                "end_date=%s: only %d/%d rows overlap between existing and new ALSTM",
                end_date, len(common_idx), len(existing),
            )
        # Build merged frame, preserving LGBM/TRA from existing
        merged = existing[lgbm_cols + tra_cols].loc[common_idx].copy()
        for c in alstm_df.columns:
            merged[c] = alstm_df[c].reindex(common_idx)
        # Re-order columns to canonical: lgbm × 3, alstm × 3, tra × 3
        ordered_cols = []
        for prefix in ("lgbm_", "alstm_", "tra_"):
            for h in ("1d", "5d", "20d"):
                col = f"{prefix}{h}"
                if col in merged.columns:
                    ordered_cols.append(col)
        merged = merged[ordered_cols]

        # Recompute ensemble score via rank_average and consensus
        rank_avg = rank_average(merged)
        merged["score"] = (-rank_avg).rename("score")
        merged["consensus"] = consensus_per_row(merged[ordered_cols])
        merged = ewma_smooth(merged, alpha=cfg.ewma_alpha, score_col="score")

        # Verify ALSTM cols populated
        for c in alstm_df.columns:
            n = merged[c].notna().sum()
            _log.info("  %s: %d/%d non-null (%.0f%%)", c, n, len(merged), 100 * n / len(merged))

        # Write back
        write_pred_pkl(merged, pred_path)
        _log.info("wrote %s shape=%s", pred_path, merged.shape)

    _log.info("done — all dates processed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
