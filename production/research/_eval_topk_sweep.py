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
import sys as _sys
import sysconfig as _sysconfig
_P = _sysconfig.get_paths().get("purelib")
if _P and _P not in _sys.path[:1]:
    _sys.path.insert(0, _P)
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pathlib import Path
import json

import numpy as np
import pandas as pd

OOF_FAC = "production/reports/oof_lgbmfac_2021_2026.pkl"
OOF_2MODEL = "production/reports/oof_2model_2021_2026.pkl"
CONFIG = "production/configs/rolling_ensemble.yaml"
PERIOD, CAPITAL, PROFILE = 5, 100_000.0, "small"
TOP_KS = [5, 10, 15, 20, 30]


def _calmar(m):
    dd, c = m.get("max_drawdown"), m.get("net_cagr")
    return (c / abs(dd)) if (dd and np.isfinite(dd) and abs(dd) > 1e-12) else float("nan")


def _pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+6.1%}"


def _num(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _daily(scores, fwd, k):
    from production.backtest.engine import run_backtest
    from production.backtest.rebalance import FixedPeriod
    from production.backtest.costs import cost_model
    return run_backtest(scores, fwd, FixedPeriod(top_k=k, period=PERIOD),
                        cost_model(PROFILE), capital=CAPITAL)["daily"]


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
    Path("logs").mkdir(exist_ok=True)
    from production.score_utils import score_of as _score_of, rebuild_2model as _rebuild_2model
    from production.backtest.metrics_net import net_metrics
    fac = pd.read_pickle(OOF_FAC)
    two = pd.read_pickle(OOF_2MODEL)
    factor_2m = _rebuild_2model(fac, two)
    base_2m = _score_of(two)

    insts = sorted(set(factor_2m.index.get_level_values("instrument")) |
                   set(base_2m.index.get_level_values("instrument")))
    dts = pd.DatetimeIndex(sorted(set(factor_2m.index.get_level_values("datetime")) |
                                  set(base_2m.index.get_level_values("datetime"))))
    start, end = str(dts.min().date()), str(dts.max().date())
    from production.backtest.data import load_fwd_returns
    fwd = load_fwd_returns(insts, start, end, config_path=CONFIG)
    years = sorted({d.year for d in dts})

    hdr = (f"{'top_k':>5} {'net_cagr':>9} {'net_ir':>7} {'max_dd':>8} {'Calmar':>7} "
           f"{'turnov':>7} {'cost/yr':>8} {'win':>6} {'neg_yrs':>7} {'t':>6} {'p':>7}")
    print("FACTOR-2MODEL — top_k concentration sweep (fixed/5d, ¥100k, small cost)")
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))

    out = {}
    per_year_rows = []
    for k in TOP_KS:
        fd = _daily(factor_2m, fwd, k)
        bd = _daily(base_2m, fwd, k)
        m = net_metrics(fd)
        cal = _calmar(m)
        # per-year
        yr = {}
        for y in years:
            sub = fd.loc[pd.to_datetime(fd.index).year == y]
            yr[y] = net_metrics(sub)["net_cagr"]
        neg = sum(1 for y in years if np.isfinite(yr[y]) and yr[y] < 0)
        t, p = _paired_t(fd["net"], bd["net"])
        out[k] = {"metrics": m, "calmar": cal, "per_year": yr, "neg_years": neg, "t": t, "p": p}
        per_year_rows.append((k, yr))
        print(f"{k:>5} {_pct(m['net_cagr']):>9} {_num(m['net_ir']):>7} {_pct(m['max_drawdown']):>8} "
              f"{_num(cal):>7} {_num(m['avg_turnover'],3):>7} {_pct(m['cost_drag_annual']):>8} "
              f"{_pct(m['win_rate']):>6} {neg:>7} {_num(t):>6} {_num(p,4):>7}")

    print("\nPER-YEAR net_cagr by top_k (factor-2model):")
    yhdr = f"{'top_k':>5} " + " ".join(f"{y:>8}" for y in years)
    print(yhdr); print("-" * len(yhdr))
    for k, yr in per_year_rows:
        print(f"{k:>5} " + " ".join(f"{_pct(yr[y]):>8}" for y in years))

    Path("logs/eval_topk_sweep_summary.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("\nwrote logs/eval_topk_sweep_summary.json")
    print("(t/p = paired daily-net t-test factor-2model vs baseline-2model at that top_k)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
