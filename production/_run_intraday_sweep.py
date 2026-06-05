"""Full P1 intraday-execution sweep: factor-2model picks, every entry rule vs the
open baseline, canonical fixed/hold-5/5d top-5, OOF 2020-07..2025-12.

The FIRST non-open rule populates the 5min + prev_close cache (slow, minutes, one
baostock session); every later rule is a cache hit. Resumable: the cache persists
per fetch, so a re-run continues where it left off.

Run from the MAIN repo:
  F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production._run_intraday_sweep > logs/intraday_sweep.log 2>&1
"""
import sys, sysconfig
_P = sysconfig.get_paths().get("purelib")
if _P and _P not in sys.path[:1]:
    sys.path.insert(0, _P)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import json
from pathlib import Path
import numpy as np
import pandas as pd

OOF_FAC = "production/reports/oof_lgbmfac_2021_2026.pkl"
OOF_2MODEL = "production/reports/oof_2model_2021_2026.pkl"

RULES = [
    ("open", {}),
    ("vwap", {}),
    ("vwap_am", {}),
    ("low_band", {"k": 0.005}),
    ("low_band", {"k": 0.01}),
    ("low_band", {"k": 0.02}),
    ("gap_cond", {"g": 0.02}),
    ("gap_cond", {"g": 0.03}),
    ("first30_low", {}),
]


def _tag(rule, p):
    if "k" in p:
        return f"{rule}(k={p['k']})"
    if "g" in p:
        return f"{rule}(g={p['g']})"
    return rule


def _pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+6.1%}"


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    from production._eval_factors import _rebuild_2model
    from production.intraday.exec_backtest import simulate

    fac = pd.read_pickle(OOF_FAC); two = pd.read_pickle(OOF_2MODEL)
    scores = _rebuild_2model(fac, two)
    dts = scores.index.get_level_values("datetime")
    print(f"score span {str(dts.min().date())}..{str(dts.max().date())} "
          f"({dts.nunique()} days)", flush=True)

    out = {}
    years = sorted({pd.Timestamp(d).year for d in dts.unique()})
    hdr = (f"{'rule':>16} {'cagr':>8} {'calmar':>7} {'maxdd':>8} {'win':>6} "
           f"{'fill':>9} {'unfl%':>6} {'fbk%':>6} {'impbp':>7} " + " ".join(f"{y:>7}" for y in years))
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for rule, p in RULES:
        tag = _tag(rule, p)
        m = simulate(scores, rule=rule, top_k=5, period=5,
                     k=p.get("k", 0.01), g=p.get("g", 0.03), cost_bps=10.0)
        out[tag] = m
        yr = m.get("by_year", {})
        print(f"{tag:>16} {_pct(m['net_cagr']):>8} {m['calmar']:>7.2f} {_pct(m['max_dd']):>8} "
              f"{_pct(m['win']):>6} {m['n_filled']:>4}/{m['n_trades']:<4} "
              f"{m['unfillable_pct']*100:>5.1f} {m['fallback_pct']*100:>5.1f} "
              f"{m['improve_bps_med']:>7.1f} "
              + " ".join(f"{_pct(yr.get(y)):>7}" for y in years), flush=True)

    base = out["open"]["net_cagr"]
    print("\n=== net_cagr delta vs open ===", flush=True)
    for tag, m in out.items():
        print(f"  {tag:>16}: {_pct(m['net_cagr'])}  (delta {m['net_cagr']-base:+.4f})", flush=True)

    Path("logs/intraday_sweep_summary.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("\nwrote logs/intraday_sweep_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
