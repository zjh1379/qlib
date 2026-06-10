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


def load_sector_map(path: str) -> pd.Series:
    """Load instrument->industry Series from a parquet with columns [instrument, industry]."""
    df = pd.read_parquet(path)
    return pd.Series(df["industry"].values, index=df["instrument"].values)


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


def _exposure_from_regime(scores: pd.Series, regime: dict, config_path: str):
    """Load the market proxy for `scores`' span and compute an exposure Series from a
    regime spec dict, e.g. {"method":"trend_ma","ma_window":60,"band":0.10}. Centralizes
    the load_market_proxy + compute_exposure wiring that callers used to repeat."""
    from .market import load_market_proxy
    from .regime import compute_exposure
    instruments = sorted(scores.index.get_level_values("instrument").unique())
    dates = scores.index.get_level_values("datetime")
    mkt = load_market_proxy(instruments, str(dates.min().date()),
                            str(dates.max().date()), config_path=config_path)
    method = regime.get("method", "trend_ma")
    exp = compute_exposure(mkt, method=method, ma_window=regime.get("ma_window", 120),
                           band=regime.get("band", 0.10),
                           min_exposure=regime.get("min_exposure", 0.0),
                           vol_target=regime.get("vol_target"))
    return exp.rename(regime.get("name") or f"{method}{regime.get('ma_window', 120)}")


def build_report(scores: pd.Series, fwd_ret: pd.Series | None = None, *,
                 policy_name: str, top_k: int, period: int, exit_k: int,
                 capital: float, profile: str, exposure=None,
                 regime: dict | None = None,
                 config_path: str = "production/configs/rolling_ensemble.yaml") -> dict:
    """Run the canonical net-of-cost backtest and return a scorecard dict.

    Deep entry point that hides the whole ritual:
      - `fwd_ret` is loaded from qlib for the scores' span if omitted;
      - `regime` (e.g. {"method":"trend_ma","ma_window":60,"band":0.10}) computes the
        exposure overlay internally when `exposure` is not passed.
    So a caller can go from the 6-step load_fwd/load_market/compute_exposure/policy/
    engine/metrics wiring to a single call. Passing `fwd_ret`/`exposure` explicitly
    still works (back-compatible)."""
    if fwd_ret is None:
        from .data import load_fwd_returns
        instruments = sorted(scores.index.get_level_values("instrument").unique())
        dates = scores.index.get_level_values("datetime")
        fwd_ret = load_fwd_returns(instruments, str(dates.min().date()),
                                   str(dates.max().date()), config_path=config_path)
    if exposure is None and regime:
        exposure = _exposure_from_regime(scores, regime, config_path)
    policy = _make_policy(policy_name, top_k, period, exit_k)
    cm = cost_model(profile)
    res = run_backtest(scores, fwd_ret, policy, cm, capital=capital, exposure=exposure)
    return {
        "params": {"policy": policy_name, "top_k": top_k, "period": period,
                   "exit_k": exit_k, "capital": capital, "profile": profile,
                   "regime": getattr(exposure, "name", None) if exposure is not None else None},
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
    ap.add_argument("--neutralize", action="store_true",
                    help="apply sector neutralization to scores before backtest")
    ap.add_argument("--industry-map", default="production/cache/industry_map.parquet")
    ap.add_argument("--regime", default="none", choices=["none", "trend_ma"])
    ap.add_argument("--ma-window", type=int, default=120)
    ap.add_argument("--band", type=float, default=0.10)
    ap.add_argument("--min-exposure", type=float, default=0.0)
    ap.add_argument("--vol-target", type=float, default=None)
    args = ap.parse_args()

    pred = pd.read_pickle(args.pred_file)
    scores = extract_score_series(pred, args.score_col).dropna()

    if args.neutralize:
        from production.neutralize import neutralize as _neutralize
        sector = load_sector_map(args.industry_map)
        scores = _neutralize(scores, sector=sector)

    regime = None
    if args.regime != "none":
        regime = {"method": args.regime, "ma_window": args.ma_window, "band": args.band,
                  "min_exposure": args.min_exposure, "vol_target": args.vol_target}

    # build_report loads fwd returns + computes the regime exposure internally.
    rep = build_report(scores, policy_name=args.policy, top_k=args.top_k,
                       period=args.period, exit_k=args.exit_k, capital=args.capital,
                       profile=args.profile, regime=regime, config_path=args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}  net_ir={rep['metrics']['net_ir']:.3f}  "
          f"turnover={rep['metrics']['avg_turnover']:.3f}  "
          f"net_cagr={rep['metrics']['net_cagr']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
