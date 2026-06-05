"""P1 intraday execution runbook. Rebuild the factor-2model champion score
(_rebuild_2model on the saved OOFs) and run the offline execution simulator under
a chosen entry rule.

  rule=open  -> REGRESSION ANCHOR: same magnitude as the canonical fixed/hold-5/5d
               +31% net engine (uses only daily $open, no 5min fetch).
  rule!=open -> fetches baostock 5min (cached) per (name, entry_date).

Run from the MAIN repo:
  F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production._run_intraday_p1 --rule open
"""
import sys, sysconfig
_P = sysconfig.get_paths().get("purelib")
if _P and _P not in sys.path[:1]:
    sys.path.insert(0, _P)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import argparse
import numpy as np
import pandas as pd

OOF_FAC = "production/reports/oof_lgbmfac_2021_2026.pkl"
OOF_2MODEL = "production/reports/oof_2model_2021_2026.pkl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="open")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--k", type=float, default=0.01)
    ap.add_argument("--g", type=float, default=0.03)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--start", default=None, help="clip scores to datetime >= start (warmup)")
    ap.add_argument("--end", default=None, help="clip scores to datetime <= end")
    args = ap.parse_args()

    from production._eval_factors import _rebuild_2model
    from production.intraday.exec_backtest import simulate

    fac = pd.read_pickle(OOF_FAC)
    two = pd.read_pickle(OOF_2MODEL)
    scores = _rebuild_2model(fac, two)
    dts = scores.index.get_level_values("datetime")
    print(f"score span: {str(dts.min().date())} .. {str(dts.max().date())} "
          f"({scores.index.get_level_values('datetime').nunique()} days, {len(scores)} rows)")
    if args.start or args.end:
        mask = np.ones(len(scores), dtype=bool)
        if args.start:
            mask &= (dts >= pd.Timestamp(args.start))
        if args.end:
            mask &= (dts <= pd.Timestamp(args.end))
        scores = scores[mask]
        print(f"clipped to {args.start}..{args.end}: {scores.index.get_level_values('datetime').nunique()} days")

    m = simulate(scores, rule=args.rule, top_k=args.top_k, period=args.period,
                 k=args.k, g=args.g, cost_bps=args.cost_bps)
    print(f"=== intraday P1 simulate [rule={args.rule}] ===")
    for key in ("rule", "net_cagr", "calmar", "max_dd", "win", "n_periods",
                "n_trades", "n_filled", "n_unfillable", "n_gap_skip", "n_no_open",
                "n_fallback", "unfillable_pct", "fallback_pct", "improve_bps_med"):
        v = m[key]
        if isinstance(v, float):
            print(f"  {key:>16}: {v:.4f}")
        else:
            print(f"  {key:>16}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
