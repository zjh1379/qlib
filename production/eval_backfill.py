"""Evaluate the pooled backfill predictions vs daily_cn_fresh.

Reads every `examples/mlruns/pred_*.pkl` file (each is one week of LightGBM ×
3-horizon rank-average ensemble), concatenates them into a unified prediction
series, fetches matching open-to-open labels from qlib, then computes the
8-metric scorecard. Prints side-by-side comparison vs daily_cn_fresh's
historical scorecard.

Usage:
  python -m production.eval_backfill
  python -m production.eval_backfill --pattern 'pred_2026-*.pkl'

Outputs:
  - stdout: side-by-side scorecard table
  - production/reports/backfill_eval_<timestamp>.json + .md
"""
from __future__ import annotations

# IMPORTANT: force conda-env qlib (compiled) over in-repo qlib/ source.
import sys as _sys
import sysconfig as _sysconfig
_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
# Append (not insert) so the in-repo qlib/ source doesn't shadow the conda
# env's compiled qlib that we put at sys.path[0] via _PURELIB.
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "backend"))


def load_pooled_pred(pattern: str) -> pd.Series:
    """Glob + concat all `pred_*.pkl` files matching pattern.
    Returns a Series of unified 'score' values indexed by (datetime, instrument)."""
    paths = sorted(glob.glob(str(REPO_ROOT / "examples" / "mlruns" / pattern)))
    if not paths:
        raise FileNotFoundError(f"no files match {pattern}")
    frames = []
    for p in paths:
        df = pd.read_pickle(p)
        if isinstance(df, pd.Series):
            df = df.to_frame(name="score")
        if "score" not in df.columns:
            if df.shape[1] == 1:
                df = df.rename(columns={df.columns[0]: "score"})
            else:
                continue
        frames.append(df[["score"]])
    if not frames:
        raise ValueError("no usable pred files (missing 'score' column)")
    pooled = pd.concat(frames, axis=0)
    pooled = pooled.dropna()
    pooled.index.names = ["datetime", "instrument"]
    # Dedup if any (date, instrument) appears in multiple weeks (keep last)
    pooled = pooled[~pooled.index.duplicated(keep="last")]
    return pooled["score"].sort_index()


def fetch_labels(pred: pd.Series) -> pd.Series:
    """Pull Ref($open, -2)/Ref($open, -1) - 1 from qlib for pred's (date, sym) range."""
    import qlib
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data_bs", region="cn")
    from qlib.data import D

    symbols = sorted(pred.index.get_level_values("instrument").unique().tolist())
    dates = pred.index.get_level_values("datetime")
    start = (pd.Timestamp(dates.min()) - pd.Timedelta(days=5)).date()
    end = (pd.Timestamp(dates.max()) + pd.Timedelta(days=10)).date()
    df = D.features(
        instruments=symbols,
        fields=["Ref($open, -2) / Ref($open, -1) - 1"],
        start_time=str(start),
        end_time=str(end),
    )
    df.columns = ["y"]
    s = df["y"]
    if s.index.names != ["datetime", "instrument"]:
        s.index.names = ["instrument", "datetime"]
        s = s.swaplevel().sort_index()
    return s


def load_daily_cn_fresh_pred() -> pd.Series:
    """Load the baseline daily_cn_fresh pred.pkl as the comparison series."""
    # Recorder f29f042f... in experiment 737988390843672861
    pred_path = REPO_ROOT / "examples" / "mlruns" / "737988390843672861" / \
        "f29f042f72634226aa0dc7782d4873d9" / "artifacts" / "pred.pkl"
    if not pred_path.exists():
        return pd.Series(dtype="float64")
    df = pd.read_pickle(pred_path)
    if isinstance(df, pd.DataFrame):
        if "score" in df.columns:
            s = df["score"]
        else:
            s = df.iloc[:, 0]
    else:
        s = df
    s.index.names = ["datetime", "instrument"]
    return s.sort_index()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="pred_*.pkl",
                        help="glob pattern under examples/mlruns/ (default: pred_*.pkl)")
    parser.add_argument("--out", default=str(REPO_ROOT / "production" / "reports"))
    args = parser.parse_args()

    from production.metrics import compute_scorecard

    print(f"Loading pooled backfill predictions: {args.pattern}", file=sys.stderr)
    pooled = load_pooled_pred(args.pattern)
    print(f"  Pooled: {len(pooled)} (date,symbol) rows, "
          f"{pooled.index.get_level_values('datetime').nunique()} dates, "
          f"{pooled.index.get_level_values('instrument').nunique()} symbols",
          file=sys.stderr)
    print(f"  Date range: {pooled.index.get_level_values('datetime').min()} → "
          f"{pooled.index.get_level_values('datetime').max()}", file=sys.stderr)

    print("Fetching open-to-open labels for backfill range...", file=sys.stderr)
    labels = fetch_labels(pooled)
    print(f"  Labels: {len(labels)} rows", file=sys.stderr)

    print("Computing backfill scorecard...", file=sys.stderr)
    sc_backfill = compute_scorecard(pooled, labels, top_k=30, bps=10)

    print("Loading daily_cn_fresh baseline + computing its scorecard...", file=sys.stderr)
    daily = load_daily_cn_fresh_pred()
    if daily.empty:
        sc_daily = {k: float("nan") for k in sc_backfill}
    else:
        labels_daily = fetch_labels(daily)
        sc_daily = compute_scorecard(daily, labels_daily, top_k=30, bps=10)

    # Print side-by-side
    print()
    print(f"{'Metric':<30} {'Backfill (β)':>14} {'daily_cn_fresh':>14} {'Δ':>10}  Threshold")
    print(f"{'-'*30} {'-'*14} {'-'*14} {'-'*10}  {'-'*15}")
    thresholds = {
        "ic_mean": ("≥ 0.030", 0.030, True),
        "ric_mean": ("—", None, True),
        "icir": ("≥ 0.40", 0.40, True),
        "top_bottom_spread_monthly": ("≥ 1.5%/mo", 1.5, True),
        "annual_excess_return": ("≥ +15%", 0.15, True),
        "ir": ("≥ 2.5", 2.5, True),
        "max_drawdown": ("≥ -15%", -0.15, True),
        "daily_turnover": ("≤ 20%", 0.20, False),
    }
    for k in sc_backfill:
        a = sc_backfill[k]
        b = sc_daily.get(k, float("nan"))
        d = a - b if pd.notna(a) and pd.notna(b) else float("nan")
        thr_text, thr_val, higher_better = thresholds.get(k, ("—", None, True))
        print(f"{k:<30} {a:>+14.4f} {b:>+14.4f} {d:>+10.4f}  {thr_text}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "computed_at": datetime.now().isoformat(),
        "backfill_pattern": args.pattern,
        "backfill_files": sorted(glob.glob(str(REPO_ROOT / "examples" / "mlruns" / args.pattern))),
        "backfill_scorecard": sc_backfill,
        "daily_cn_fresh_scorecard": sc_daily,
        "backfill_meta": {
            "rows": int(len(pooled)),
            "dates": pooled.index.get_level_values("datetime").nunique(),
            "symbols": pooled.index.get_level_values("instrument").nunique(),
            "start": str(pooled.index.get_level_values("datetime").min()),
            "end": str(pooled.index.get_level_values("datetime").max()),
        },
    }
    json_path = out_dir / f"backfill_eval_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
