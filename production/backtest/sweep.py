"""Grid sweep over policy x top_k x holding-period x capital -> net-return frontier.

CLI:
  F:/Tools/Anaconda/envs/qlib/python.exe -m production.backtest.sweep \
    --pred-file <pred.pkl> --out production/reports/sweep.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .costs import cost_model
from .engine import run_backtest
from .metrics_net import net_metrics
from .rebalance import Daily, FixedPeriod, Banded


def _policy(name: str, top_k: int, period: int):
    if name == "daily":
        return Daily(top_k=top_k)
    if name == "fixed":
        return FixedPeriod(top_k=top_k, period=period)
    if name == "banded":
        return Banded(top_k=top_k, exit_k=2 * top_k)
    raise ValueError(name)


def run_sweep(scores: pd.Series, fwd_ret: pd.Series, *, policies, top_ks,
              periods, capitals, profile: str = "small") -> pd.DataFrame:
    cm = cost_model(profile)
    rows = []
    for pol in policies:
        # period only varies the grid for the 'fixed' policy
        per_list = periods if pol == "fixed" else [periods[0]]
        for k in top_ks:
            for per in per_list:
                for cap in capitals:
                    res = run_backtest(scores, fwd_ret, _policy(pol, k, per), cm, capital=cap)
                    m = net_metrics(res["daily"])
                    rows.append({"policy": pol, "top_k": k, "period": per,
                                 "capital": cap, **m})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest parameter sweep.")
    ap.add_argument("--pred-file", required=True)
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--profile", default="small", choices=["small", "pro"])
    ap.add_argument("--config", default="production/configs/rolling_ensemble.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from .run import extract_score_series
    from .data import load_fwd_returns

    pred = pd.read_pickle(args.pred_file)
    scores = extract_score_series(pred, args.score_col).dropna()
    instruments = sorted(scores.index.get_level_values("instrument").unique())
    dates = scores.index.get_level_values("datetime")
    fwd = load_fwd_returns(instruments, str(dates.min().date()), str(dates.max().date()),
                           config_path=args.config)

    grid = run_sweep(scores, fwd,
                     policies=["daily", "fixed", "banded"],
                     top_ks=[5, 10, 15, 20, 30],
                     periods=[1, 2, 3, 5],
                     capitals=[50_000, 100_000, 300_000, 1_000_000],
                     profile=args.profile)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.sort_values("net_ir", ascending=False).to_csv(out, index=False, encoding="utf-8-sig")
    best = grid.loc[grid["net_ir"].idxmax()]
    print(f"wrote {out} rows={len(grid)}")
    print(f"BEST net_ir: policy={best['policy']} top_k={best['top_k']} "
          f"capital={best['capital']} net_ir={best['net_ir']:.3f} "
          f"turnover={best['avg_turnover']:.3f} net_cagr={best['net_cagr']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
