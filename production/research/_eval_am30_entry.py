# production/research/_eval_am30_entry.py
"""Exact first-30-min entry penalty on the deployable top-3: simulate open vs
am30_vwap (first-30-min VWAP) entry, exit = next-open (P1 framework), period=5
(reuses P1's cached 5min — top-3/period-5 entries are a subset of P1's top-5 set).
Quantifies precisely what the user's 'buy within the first 30 min' costs vs the open.

Run: F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._eval_am30_entry \
  > logs/eval_am30_entry.log 2>&1
"""
from production.research._harness import bootstrap, OOF_FAC, OOF_2MODEL, champion_scores
bootstrap()

import json
from pathlib import Path
import pandas as pd


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    from production.intraday.exec_backtest import simulate
    scores = champion_scores()
    out = {}
    print(f"top-3, exit=next-open, period=5, 30bp/rt")
    print(f"{'rule':>10} {'net_cagr':>9} {'Calmar':>7} {'maxDD':>8} {'win':>6} "
          f"{'fill':>6} {'unfl%':>6} {'fbk%':>6}")
    print("-" * 66)
    base = None
    for rule in ["open", "am30_vwap"]:
        r = simulate(scores, rule=rule, top_k=3, period=5, cost_bps=30.0)
        out[rule] = r
        if rule == "open":
            base = r["net_cagr"]
        delta = "" if rule == "open" else f"  (Δ {(r['net_cagr'] - base) * 100:+.1f}pp)"
        print(f"{rule:>10} {r['net_cagr']:>+9.2%} {r['calmar']:>7.2f} {r['max_dd']:>+8.2%} "
              f"{r['win']:>6.1%} {r['n_filled']:>6} {r['unfillable_pct']:>6.1%} "
              f"{r['fallback_pct']:>6.1%}{delta}")
    Path("logs/eval_am30_entry.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("\nwrote logs/eval_am30_entry.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
