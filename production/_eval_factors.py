"""P2b factor incremental evaluation: does adding the 6 short-term factors to
LGBM's Alpha158 thicken alpha? Compares, under the CANONICAL net-of-cost backtest
(fixed / hold-5 / 5-day, ¥100k, small cost) — same config as the +19% baseline:

  A. factor-LGBM-only      vs  baseline-LGBM-only        (oof_lgbmfac vs oof_lgbm)
  B. factor-LGBM + ALSTM   vs  baseline-LGBM + ALSTM      (rebuilt vs oof_2model = +19%)
  C. both B variants WITH the P3 exposure overlay (ma60/band0.10)

Run AFTER the factor backfill completes:
  F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production._pool_fac
  F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production._eval_factors > logs/eval_factors.log 2>&1
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

from production.score_utils import (
    score_of as _score_of, rebuild_2model as _rebuild_2model, calmar as _calmar)

OOF_FAC = "production/reports/oof_lgbmfac_2021_2026.pkl"     # factor LGBM (pooled)
OOF_LGBM = "production/reports/oof_lgbm_2021_2026.pkl"       # baseline LGBM
OOF_2MODEL = "production/reports/oof_2model_2021_2026.pkl"   # baseline LGBM+ALSTM (=+19%)
CONFIG = "production/configs/rolling_ensemble.yaml"

TOP_K, PERIOD, CAPITAL, PROFILE = 5, 5, 100_000.0, "small"
OVERLAY_MA, OVERLAY_BAND = 60, 0.10


def _fmt_pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+7.2%}"


def _fmt_num(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _run(name, scores, fwd, exposure, regimes_out):
    from production.backtest.run import build_report
    rep = build_report(scores, fwd, policy_name="fixed", top_k=TOP_K, period=PERIOD,
                       exit_k=2 * TOP_K, capital=CAPITAL, profile=PROFILE, exposure=exposure)
    m = rep["metrics"]
    regimes_out[name] = rep["regimes"]
    return {"variant": name, "net_cagr": m["net_cagr"], "net_ir": m["net_ir"],
            "avg_turnover": m["avg_turnover"], "max_drawdown": m["max_drawdown"],
            "calmar": _calmar(m), "win_rate": m["win_rate"], "n_days": m["n_days"],
            "final_nav": rep["final_nav"]}


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    for p in (OOF_FAC, OOF_LGBM, OOF_2MODEL):
        if not Path(p).exists():
            print(f"MISSING: {p} — run the factor backfill + _pool_fac first.")
            return 1

    fac = pd.read_pickle(OOF_FAC)
    lgbm = pd.read_pickle(OOF_LGBM)
    two = pd.read_pickle(OOF_2MODEL)
    print("OOF columns:")
    print(f"  factor-lgbm : {list(fac.columns)}  rows={len(fac):,}")
    print(f"  base-lgbm   : {list(lgbm.columns)}  rows={len(lgbm):,}")
    print(f"  base-2model : {list(two.columns)}  rows={len(two):,}")

    scores = {
        "baseline-LGBM": _score_of(lgbm),
        "factor-LGBM": _score_of(fac),
        "baseline-2model(+19%)": _score_of(two),
        "factor-LGBM+ALSTM": _rebuild_2model(fac, two),
    }

    # forward returns + overlay over the union span/instruments (load once)
    all_inst = sorted(set().union(*[set(s.index.get_level_values("instrument")) for s in scores.values()]))
    all_dt = pd.DatetimeIndex(sorted(set().union(*[set(s.index.get_level_values("datetime")) for s in scores.values()])))
    start, end = str(all_dt.min().date()), str(all_dt.max().date())
    print(f"\nunion: {len(all_inst)} instruments | {len(all_dt)} dates | {start} .. {end}")

    from production.backtest.data import load_fwd_returns
    fwd = load_fwd_returns(all_inst, start, end, config_path=CONFIG)
    print(f"fwd returns: {len(fwd):,} rows")

    from production.backtest.market import load_market_proxy
    from production.backtest.regime import compute_exposure
    mkt = load_market_proxy(all_inst, start, end, config_path=CONFIG)
    exposure = compute_exposure(mkt, method="trend_ma", ma_window=OVERLAY_MA, band=OVERLAY_BAND).rename("trend_ma60")

    regimes = {}
    rows = []
    # A: LGBM-only factor delta
    rows.append(_run("baseline-LGBM", scores["baseline-LGBM"], fwd, None, regimes))
    rows.append(_run("factor-LGBM", scores["factor-LGBM"], fwd, None, regimes))
    # B: 2-model factor delta
    rows.append(_run("baseline-2model(+19%)", scores["baseline-2model(+19%)"], fwd, None, regimes))
    rows.append(_run("factor-LGBM+ALSTM", scores["factor-LGBM+ALSTM"], fwd, None, regimes))
    # C: + overlay on the 2-model variants
    rows.append(_run("baseline-2model +overlay", scores["baseline-2model(+19%)"], fwd, exposure, regimes))
    rows.append(_run("factor-LGBM+ALSTM +overlay", scores["factor-LGBM+ALSTM"], fwd, exposure, regimes))

    hdr = (f"{'variant':<28} {'net_cagr':>9} {'net_ir':>7} {'turnover':>9} "
           f"{'max_dd':>9} {'Calmar':>7} {'win':>7} {'days':>5}")
    print("\n" + "=" * len(hdr))
    print("FACTOR INCREMENTAL EVAL (policy=fixed top_k=5 period=5, ¥100k, small) [canonical]")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['variant']:<28} {_fmt_pct(r['net_cagr']):>9} {_fmt_num(r['net_ir']):>7} "
              f"{_fmt_num(r['avg_turnover'], 3):>9} {_fmt_pct(r['max_drawdown']):>9} "
              f"{_fmt_num(r['calmar']):>7} {_fmt_pct(r['win_rate']):>7} {r['n_days']:>5}")

    # per-regime net_cagr
    keys = []
    for nm in regimes:
        for k in regimes[nm]:
            if k not in keys:
                keys.append(k)
    keys.sort()
    print("\nPer-regime net_cagr:")
    h2 = f"{'variant':<28} " + " ".join(f"{k.split('__')[0][:7]:>9}" for k in keys)
    print(h2)
    print("-" * len(h2))
    for r in rows:
        cells = [_fmt_pct(regimes[r['variant']].get(k, {}).get("net_cagr")) for k in keys]
        print(f"{r['variant']:<28} " + " ".join(f"{c:>9}" for c in cells))

    # deltas
    def g(n):
        return next(x for x in rows if x["variant"] == n)
    print("\nDELTAS (factor − baseline):")
    dl1 = g("factor-LGBM")["net_cagr"] - g("baseline-LGBM")["net_cagr"]
    di1 = g("factor-LGBM")["net_ir"] - g("baseline-LGBM")["net_ir"]
    dl2 = g("factor-LGBM+ALSTM")["net_cagr"] - g("baseline-2model(+19%)")["net_cagr"]
    di2 = g("factor-LGBM+ALSTM")["net_ir"] - g("baseline-2model(+19%)")["net_ir"]
    print(f"  LGBM-only : net_cagr {dl1:+.2%}  net_ir {di1:+.3f}")
    print(f"  2-model   : net_cagr {dl2:+.2%}  net_ir {di2:+.3f}")

    Path("logs/eval_factors_summary.json").write_text(
        json.dumps(rows, indent=2, default=float), encoding="utf-8")
    print("\nwrote logs/eval_factors_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
