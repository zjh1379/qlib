# production/research/_eval_executability.py
"""涨停 可成交性 + 选择性偏差 (Block 2). Ungated vs gated (skip names whose entry
open gaps to the 涨停 ceiling) net metrics at top_k {1,2,3,5}/¥10k, plus the
buyable-vs-unbuyable realized-return split (does the live trader miss the winners?).

Run: F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._eval_executability \
  > logs/eval_executability.log 2>&1
"""
from production.research._harness import bootstrap, OOF_FAC, OOF_2MODEL, CONFIG, champion_scores
bootstrap()

import json
from pathlib import Path
import pandas as pd
PERIOD, CAPITAL, PROFILE = 5, 10_000.0, "small"
TOP_KS = [1, 2, 3, 5]


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    from production.backtest.data import load_fwd_returns
    from production.backtest.executability import (
        load_entry_ohlc, buyable_mask, gate_scores, selection_bias_split)
    from production.backtest.run import build_report

    scores = champion_scores()
    insts = sorted(scores.index.get_level_values("instrument").unique())
    dts = scores.index.get_level_values("datetime")
    start, end = str(dts.min().date()), str(dts.max().date())

    fwd = load_fwd_returns(insts, start, end, config_path=CONFIG)
    ohlc = load_entry_ohlc(insts, start, end, config_path=CONFIG)
    buyable = buyable_mask(ohlc)
    gated = gate_scores(scores, buyable)

    out = {"ungated": {}, "gated": {}, "bias": {}}
    print(f"{'top_k':>5} {'ungated':>9} {'gated':>9} {'unbuy%':>7} {'edge_missed':>12}")
    print("-" * 46)
    for k in TOP_KS:
        u = build_report(scores, fwd_ret=fwd, policy_name="fixed", top_k=k, period=PERIOD,
                         exit_k=2 * k, capital=CAPITAL, profile=PROFILE, config_path=CONFIG)
        g = build_report(gated, fwd_ret=fwd, policy_name="fixed", top_k=k, period=PERIOD,
                         exit_k=2 * k, capital=CAPITAL, profile=PROFILE, config_path=CONFIG)
        bias = selection_bias_split(scores, fwd, buyable, top_k=k, period=PERIOD)
        out["ungated"][k] = u["metrics"]; out["gated"][k] = g["metrics"]; out["bias"][k] = bias
        print(f"{k:>5} {u['metrics']['net_cagr']:>+9.2%} {g['metrics']['net_cagr']:>+9.2%} "
              f"{bias['unbuyable_pct']:>7.1%} {bias['edge_missed']:>+12.2%}")
    Path("logs/eval_executability.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("wrote logs/eval_executability.json")
    print("(edge_missed>0 => the unbuyable gapped-up picks are the winners you can't buy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
