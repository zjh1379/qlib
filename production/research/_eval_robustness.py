"""Robustness / credibility check for the factor-2model's headline +31% net CAGR.

Answers three questions, all under the CANONICAL config (fixed/hold-5/5d, ¥100k,
small cost), reusing the same OOF score construction as _eval_factors:

  (1) Significance — paired t-test of daily net returns: factor-2model vs
      baseline-2model, and factor-LGBM vs baseline-LGBM. Is the +12pp real or noise?
  (2) Ex-2020 — strip the COVID-bounce window (keep dates >= 2021-01-01) and see
      how much net CAGR / Calmar survives.
  (3) Per-year stability — net CAGR + net_ir by calendar year.

Run: F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._eval_robustness > logs/eval_robustness.log 2>&1
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
OOF_LGBM = "production/reports/oof_lgbm_2021_2026.pkl"
OOF_2MODEL = "production/reports/oof_2model_2021_2026.pkl"
CONFIG = "production/configs/rolling_ensemble.yaml"
TOP_K, PERIOD, CAPITAL, PROFILE = 5, 5, 100_000.0, "small"
EX2020_START = "2021-01-01"


def _calmar(m):
    dd, cagr = m.get("max_drawdown"), m.get("net_cagr")
    if dd is None or not np.isfinite(dd) or abs(dd) < 1e-12:
        return float("nan")
    return cagr / abs(dd)


def _pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+7.2%}"


def _num(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _backtest_daily(scores, fwd):
    from production.backtest.engine import run_backtest
    from production.backtest.rebalance import FixedPeriod
    from production.backtest.costs import cost_model
    res = run_backtest(scores, fwd, FixedPeriod(top_k=TOP_K, period=PERIOD),
                       cost_model(PROFILE), capital=CAPITAL)
    return res["daily"]


def _metrics(daily):
    from production.backtest.metrics_net import net_metrics
    m = net_metrics(daily)
    m["calmar"] = _calmar(m)
    return m


def _paired_t(a: pd.Series, b: pd.Series):
    """Paired t-test of daily net returns a (treatment) vs b (control)."""
    idx = a.index.intersection(b.index)
    d = (a.reindex(idx) - b.reindex(idx)).dropna()
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return {"n": n, "mean_daily": float(d.mean()) if n else float("nan"),
                "t": float("nan"), "p": float("nan")}
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
    except Exception:
        # normal approx if scipy missing
        from math import erfc, sqrt
        p = float(erfc(abs(t) / sqrt(2)))
    return {"n": n, "mean_daily": float(d.mean()),
            "mean_annual": float(d.mean() * 252), "t": t, "p": p}


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    for p in (OOF_FAC, OOF_LGBM, OOF_2MODEL):
        if not Path(p).exists():
            print(f"MISSING {p}"); return 1

    from production.score_utils import score_of as _score_of, rebuild_2model as _rebuild_2model
    fac = pd.read_pickle(OOF_FAC)
    lgbm = pd.read_pickle(OOF_LGBM)
    two = pd.read_pickle(OOF_2MODEL)

    scores = {
        "baseline-LGBM": _score_of(lgbm),
        "factor-LGBM": _score_of(fac),
        "baseline-2model": _score_of(two),
        "factor-2model": _rebuild_2model(fac, two),
    }
    all_inst = sorted(set().union(*[set(s.index.get_level_values("instrument")) for s in scores.values()]))
    dts = pd.DatetimeIndex(sorted(set().union(*[set(s.index.get_level_values("datetime")) for s in scores.values()])))
    start, end = str(dts.min().date()), str(dts.max().date())
    print(f"universe={len(all_inst)} dates={len(dts)} span={start}..{end}")

    from production.backtest.data import load_fwd_returns
    fwd = load_fwd_returns(all_inst, start, end, config_path=CONFIG)

    # one backtest per model -> daily ledger
    daily = {name: _backtest_daily(s, fwd) for name, s in scores.items()}

    # ---- (1) + (2): full vs ex-2020 metrics ----
    def _slice(d, start_date=None):
        if start_date is None:
            return d
        return d.loc[pd.to_datetime(d.index) >= pd.Timestamp(start_date)]

    print("\n" + "=" * 92)
    print("FULL PERIOD vs EX-2020 (fixed/hold-5/5d, ¥100k, small cost)")
    print("=" * 92)
    hdr = f"{'model':<18} {'period':<10} {'net_cagr':>9} {'net_ir':>7} {'max_dd':>9} {'Calmar':>7} {'days':>5}"
    print(hdr); print("-" * len(hdr))
    rows_out = {}
    for name, d in daily.items():
        for label, sd in (("full", None), ("ex-2020", EX2020_START)):
            m = _metrics(_slice(d, sd))
            rows_out[f"{name}|{label}"] = m
            print(f"{name:<18} {label:<10} {_pct(m['net_cagr']):>9} {_num(m['net_ir']):>7} "
                  f"{_pct(m['max_drawdown']):>9} {_num(m['calmar']):>7} {m['n_days']:>5}")

    # ---- (3): per-year net_cagr / net_ir ----
    years = sorted({pd.Timestamp(d).year for d in daily["factor-2model"].index})
    print("\n" + "=" * 92)
    print("PER-YEAR net_cagr (net_ir)")
    print("=" * 92)
    yhdr = f"{'model':<18} " + " ".join(f"{y:>14}" for y in years)
    print(yhdr); print("-" * len(yhdr))
    per_year = {}
    for name, d in daily.items():
        cells = []
        per_year[name] = {}
        for y in years:
            sub = d.loc[pd.to_datetime(d.index).year == y]
            m = _metrics(sub)
            per_year[name][y] = {"net_cagr": m["net_cagr"], "net_ir": m["net_ir"]}
            cells.append(f"{_pct(m['net_cagr'])}({_num(m['net_ir'],1)})")
        print(f"{name:<18} " + " ".join(f"{c:>14}" for c in cells))

    # ---- (1): paired t-tests on daily net returns ----
    print("\n" + "=" * 92)
    print("PAIRED t-TEST on daily net returns (treatment vs control)")
    print("=" * 92)
    tests = {
        "factor-2model vs baseline-2model": ("factor-2model", "baseline-2model"),
        "factor-LGBM vs baseline-LGBM": ("factor-LGBM", "baseline-LGBM"),
        "factor-2model vs factor-LGBM (ALSTM lift)": ("factor-2model", "factor-LGBM"),
    }
    tt_out = {}
    for label, (a, b) in tests.items():
        res = _paired_t(daily[a]["net"], daily[b]["net"])
        tt_out[label] = res
        sig = "***" if res["p"] < 0.01 else "**" if res["p"] < 0.05 else "*" if res["p"] < 0.10 else "ns"
        print(f"  {label:<44} n={res['n']:>4}  mean_daily={res['mean_daily']:+.5f} "
              f"(~{res.get('mean_annual', float('nan'))*1:+.1%}/yr)  t={_num(res['t'])}  p={_num(res['p'],4)}  [{sig}]")

    Path("logs/eval_robustness_summary.json").write_text(
        json.dumps({"metrics": rows_out, "per_year": per_year, "ttests": tt_out},
                   indent=2, default=float), encoding="utf-8")
    print("\nwrote logs/eval_robustness_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
