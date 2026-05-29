"""CLI: evaluate a prediction file under a rebalance policy + realistic costs,
write a net-of-cost JSON scorecard. Usage:

  F:/Tools/Anaconda/envs/qlib/python.exe -m production.backtest.run \
    --pred-file examples/mlruns/.../pred_2026-05-22.pkl \
    --policy banded --top-k 15 --exit-k 30 --capital 100000 \
    --out production/reports/backtest_banded.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .costs import cost_model
from .engine import run_backtest
from .metrics_net import net_metrics, net_regime
from .rebalance import Daily, FixedPeriod, Banded


def extract_score_series(pred, score_col: str = "score") -> pd.Series:
    if isinstance(pred, pd.DataFrame):
        col = score_col if score_col in pred.columns else pred.columns[0]
        return pred[col]
    return pred


def _make_policy(name: str, top_k: int, period: int, exit_k: int):
    if name == "daily":
        return Daily(top_k=top_k)
    if name == "fixed":
        return FixedPeriod(top_k=top_k, period=period)
    if name == "banded":
        return Banded(top_k=top_k, exit_k=exit_k)
    raise ValueError(f"unknown policy {name!r}")


def build_report(scores: pd.Series, fwd_ret: pd.Series, *, policy_name: str,
                 top_k: int, period: int, exit_k: int, capital: float,
                 profile: str) -> dict:
    policy = _make_policy(policy_name, top_k, period, exit_k)
    cm = cost_model(profile)
    res = run_backtest(scores, fwd_ret, policy, cm, capital=capital)
    return {
        "params": {"policy": policy_name, "top_k": top_k, "period": period,
                   "exit_k": exit_k, "capital": capital, "profile": profile},
        "metrics": net_metrics(res["daily"]),
        "regimes": net_regime(res["daily"]),
        "final_nav": res["final_nav"],
        "generated_at": datetime.utcnow().isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Net-of-cost backtest of a prediction file.")
    ap.add_argument("--pred-file", required=True)
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--policy", default="daily", choices=["daily", "fixed", "banded"])
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--exit-k", type=int, default=60)
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--profile", default="small", choices=["small", "pro"])
    ap.add_argument("--config", default="production/configs/rolling_ensemble.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred = pd.read_pickle(args.pred_file)
    scores = extract_score_series(pred, args.score_col).dropna()
    instruments = sorted(scores.index.get_level_values("instrument").unique())
    dates = scores.index.get_level_values("datetime")
    start, end = str(dates.min().date()), str(dates.max().date())

    from .data import load_fwd_returns
    fwd = load_fwd_returns(instruments, start, end, config_path=args.config)

    rep = build_report(scores, fwd, policy_name=args.policy, top_k=args.top_k,
                       period=args.period, exit_k=args.exit_k, capital=args.capital,
                       profile=args.profile)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}  net_ir={rep['metrics']['net_ir']:.3f}  "
          f"turnover={rep['metrics']['avg_turnover']:.3f}  "
          f"net_cagr={rep['metrics']['net_cagr']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
