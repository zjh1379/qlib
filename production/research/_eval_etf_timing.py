# production/research/_eval_etf_timing.py
"""ETF market-timing arm: hold a broad ETF proxy, time it with the P3 trend overlay,
compare net to the single-stock sweep. v1 ETF proxy = synthetic equal-weight universe
(market.py); refinement = real index/ETF NAV.

Run: F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._eval_etf_timing \
  > logs/eval_etf_timing.log 2>&1
"""
from production.research._harness import bootstrap, OOF_2MODEL, CONFIG
bootstrap()

import json
from pathlib import Path
import pandas as pd
CAPITAL = 10_000.0
MA_WINDOW, BAND = 60, 0.10


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    from production.score_utils import score_of
    from production.backtest.market import load_market_proxy
    from production.backtest.regime import compute_exposure
    from production.backtest.etf_timing import simulate_etf_timing
    from production.backtest.costs import cost_model
    from production.backtest.metrics_net import net_metrics, tail_stats

    two = pd.read_pickle(OOF_2MODEL)
    s = score_of(two)
    insts = sorted(s.index.get_level_values("instrument").unique())
    dts = s.index.get_level_values("datetime")
    start, end = str(dts.min().date()), str(dts.max().date())

    mkt_close = load_market_proxy(insts, start, end, config_path=CONFIG)
    etf_ret = mkt_close.pct_change().dropna()
    exposure = compute_exposure(mkt_close, method="trend_ma", ma_window=MA_WINDOW, band=BAND)

    out = {}
    bh = simulate_etf_timing(etf_ret, pd.Series(1.0, index=etf_ret.index),
                             cost_model("etf"), capital=CAPITAL)
    tm = simulate_etf_timing(etf_ret, exposure, cost_model("etf"), capital=CAPITAL)
    for name, led in [("etf_buyhold", bh), ("etf_timed", tm)]:
        m = net_metrics(led)
        ts = tail_stats(led["net"])
        out[name] = {"metrics": m, "tail": ts}
        print(f"{name:>12}: net_cagr={m['net_cagr']:+.2%} maxDD={m['max_drawdown']:+.2%} "
              f"IR={m['net_ir']:.2f} neg_day%={ts['neg_period_pct']:.1%}")
    Path("logs/eval_etf_timing.json").write_text(json.dumps(out, indent=2, default=float),
                                                 encoding="utf-8")
    print("wrote logs/eval_etf_timing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
