# production/research/_eval_deploy.py
"""A2b: find the best DEPLOYABLE single-stock config at ¥10k, now that the ETF path
is dead (real broad indices ~0-3%/yr over 2020-25, see _eval_etf_real). Matrix over
top_k {1,2,3} x {plain, P3 trend overlay} x {ungated, 涨停-gated}, fixed/5d, ¥10k,
small cost. Goal: cut the -55~-90% single-stock drawdown while keeping the reversal edge.

Run: F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._eval_deploy \
  > logs/eval_deploy.log 2>&1
"""
from production.research._harness import bootstrap, OOF_FAC, OOF_2MODEL, CONFIG, champion_scores
bootstrap()

import json
from pathlib import Path
import pandas as pd
PERIOD, CAPITAL, PROFILE = 5, 10_000.0, "small"
TOP_KS = [1, 2, 3]
OVERLAY = {"method": "trend_ma", "ma_window": 60, "band": 0.10}


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    from production.backtest.data import load_fwd_returns
    from production.backtest.executability import load_entry_ohlc, buyable_mask, gate_scores
    from production.backtest.run import build_report

    scores = champion_scores()
    insts = sorted(scores.index.get_level_values("instrument").unique())
    dts = scores.index.get_level_values("datetime")
    start, end = str(dts.min().date()), str(dts.max().date())
    fwd = load_fwd_returns(insts, start, end, config_path=CONFIG)
    ohlc = load_entry_ohlc(insts, start, end, config_path=CONFIG)
    buyable = buyable_mask(ohlc)
    gated = gate_scores(scores, buyable)

    def cal(m):
        dd = m.get("max_drawdown")
        return (m["net_cagr"] / abs(dd)) if dd and abs(dd) > 1e-9 else float("nan")

    out = {}
    print(f"{'top_k':>5} {'gate':>8} {'overlay':>8} {'net_cagr':>9} {'maxDD':>8} {'Calmar':>7} {'IR':>6}")
    print("-" * 56)
    for k in TOP_KS:
        for gtag, sc in [("ungated", scores), ("gated", gated)]:
            for otag, rg in [("plain", None), ("overlay", OVERLAY)]:
                rep = build_report(sc, fwd_ret=fwd, policy_name="fixed", top_k=k, period=PERIOD,
                                   exit_k=2 * k, capital=CAPITAL, profile=PROFILE,
                                   regime=rg, config_path=CONFIG)
                m = rep["metrics"]
                out[f"k{k}_{gtag}_{otag}"] = m
                print(f"{k:>5} {gtag:>8} {otag:>8} {m['net_cagr']:>+9.2%} "
                      f"{m['max_drawdown']:>+8.2%} {cal(m):>7.2f} {m['net_ir']:>6.2f}")
    Path("logs/eval_deploy.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("\nwrote logs/eval_deploy.json  (key cells: gated x overlay for k=2,3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
