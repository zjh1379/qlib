"""top_k concentration sweep for the factor-2model (no retrain).

Robustness check (2026-06-04-robustness-results.md) found the +31% is real but
statistically marginal (p~0.10) and has 2 negative years — root cause = 5-name
over-concentration. This sweeps top_k in {5,10,15,20,30} under the canonical
fixed/5-day backtest (¥100k, small cost) and asks, for each k:
  - factor-2model net_cagr / net_ir / maxDD / Calmar / turnover / cost_drag
  - per-year net_cagr (does spreading fix 2022/2023?)
  - paired t-test factor-2model vs baseline-2model daily net (does it tighten?)

NOTE: at ¥100k the min ¥5 commission bites more as positions shrink (k up ->
smaller positions), so cost_drag is reported — the NET answer includes it.

Run: F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._eval_topk_sweep > logs/eval_topk_sweep.log 2>&1
"""
from production.research._harness import bootstrap, OOF_FAC, OOF_2MODEL, CONFIG, champion_scores, pct, num
bootstrap()

from pathlib import Path
import json

import numpy as np
import pandas as pd
from production.score_utils import calmar

PERIOD, PROFILE = 5, "small"
CAPITALS = [10_000.0, 100_000.0]
TOP_KS = [1, 2, 3, 5, 10, 15, 20, 30]


def _daily(scores, fwd, k, capital):
    from production.backtest.engine import run_backtest
    from production.backtest.rebalance import FixedPeriod
    from production.backtest.costs import cost_model
    return run_backtest(scores, fwd, FixedPeriod(top_k=k, period=PERIOD),
                        cost_model(PROFILE), capital=capital)["daily"]


def _paired_t(a, b):
    idx = a.index.intersection(b.index)
    d = (a.reindex(idx) - b.reindex(idx)).dropna()
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float("nan"), float("nan")
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
    except Exception:
        from math import erfc, sqrt
        p = float(erfc(abs(t) / sqrt(2)))
    return t, p


def main() -> int:
    from production.score_utils import score_of as _score_of
    from production.backtest.metrics_net import net_metrics, tail_stats
    factor_2m = champion_scores()
    two = pd.read_pickle(OOF_2MODEL)
    base_2m = _score_of(two)

    insts = sorted(set(factor_2m.index.get_level_values("instrument")) |
                   set(base_2m.index.get_level_values("instrument")))
    dts = pd.DatetimeIndex(sorted(set(factor_2m.index.get_level_values("datetime")) |
                                  set(base_2m.index.get_level_values("datetime"))))
    start, end = str(dts.min().date()), str(dts.max().date())
    from production.backtest.data import load_fwd_returns
    fwd = load_fwd_returns(insts, start, end, config_path=CONFIG)
    years = sorted({d.year for d in dts})

    out = {}
    for capital in CAPITALS:
        hdr = (f"{'top_k':>5} {'net_cagr':>9} {'net_ir':>7} {'max_dd':>8} {'Calmar':>7} "
               f"{'turnov':>7} {'cost/yr':>8} {'win':>6} {'p10':>8} {'std':>7} {'neg%':>6} "
               f"{'neg_yr':>6} {'t':>6} {'p':>7}")
        print(f"\nFACTOR-2MODEL — top_k sweep (fixed/5d, Y{capital:,.0f}, {PROFILE} cost)")
        print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
        cap_out = {}
        for k in TOP_KS:
            fd = _daily(factor_2m, fwd, k, capital)
            bd = _daily(base_2m, fwd, k, capital)
            m = net_metrics(fd)
            cal = calmar(m)
            ts = tail_stats(fd["net"])
            yr = {}
            for y in years:
                sub = fd.loc[pd.to_datetime(fd.index).year == y]
                yr[y] = net_metrics(sub)["net_cagr"]
            neg = sum(1 for y in years if np.isfinite(yr[y]) and yr[y] < 0)
            t, p = _paired_t(fd["net"], bd["net"])
            cap_out[k] = {"metrics": m, "calmar": cal, "tail": ts, "per_year": yr,
                          "neg_years": neg, "t": t, "p": p}
            print(f"{k:>5} {pct(m['net_cagr']):>9} {num(m['net_ir']):>7} "
                  f"{pct(m['max_drawdown']):>8} {num(cal):>7} {num(m['avg_turnover'],3):>7} "
                  f"{pct(m['cost_drag_annual']):>8} {pct(m['win_rate']):>6} "
                  f"{pct(ts['ret_p10']):>8} {num(ts['ret_std'],4):>7} "
                  f"{pct(ts['neg_period_pct']):>6} {neg:>6} {num(t):>6} {num(p,4):>7}")
        out[f"capital_{int(capital)}"] = cap_out

    Path("logs/eval_topk_sweep_summary.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("\nwrote logs/eval_topk_sweep_summary.json")
    print("(p10/std/neg% = per-DAY net left-tail; t/p = paired daily-net vs baseline at that k)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
