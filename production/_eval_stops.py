"""Per-name stop-loss sweep on the factor-2model champion (no retrain).

Question: does a per-name stop-loss cut the 2022/2023 losses + the -41% drawdown
and improve Calmar, without giving up too much return? Sweeps stop_loss levels
under the canonical fixed/hold-5/5d net-of-cost backtest (¥100k, small cost),
both without and with the P3 exposure overlay.

Run AFTER the factor OOF exists:
  python -X utf8 -m production._eval_stops > logs/eval_stops.log 2>&1
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
TOP_K, PERIOD, CAPITAL, PROFILE = 5, 5, 100_000.0, "small"
STOPS = [None, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
OVERLAY_MA, OVERLAY_BAND = 60, 0.10


def _pct(x): return "n/a" if x is None or not np.isfinite(x) else f"{x:+6.1%}"
def _num(x, n=2): return "n/a" if x is None or not np.isfinite(x) else f"{x:.{n}f}"
def _calmar(m):
    dd, c = m.get("max_drawdown"), m.get("net_cagr")
    return (c / abs(dd)) if (dd and np.isfinite(dd) and abs(dd) > 1e-12) else float("nan")


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    from production._eval_factors import _rebuild_2model
    from production.backtest.metrics_net import net_metrics
    from production.backtest.engine import run_backtest
    from production.backtest.rebalance import FixedPeriod
    from production.backtest.costs import cost_model
    from production.backtest.data import load_fwd_returns
    from production.backtest.market import load_market_proxy
    from production.backtest.regime import compute_exposure

    fac = pd.read_pickle(OOF_FAC)
    two = pd.read_pickle(OOF_2MODEL)
    scores = _rebuild_2model(fac, two)            # factor-2model champion
    insts = sorted(scores.index.get_level_values("instrument").unique())
    dts = scores.index.get_level_values("datetime")
    start, end = str(dts.min().date()), str(dts.max().date())
    fwd = load_fwd_returns(insts, start, end, config_path=CONFIG)
    mkt = load_market_proxy(insts, start, end, config_path=CONFIG)
    exposure = compute_exposure(mkt, method="trend_ma", ma_window=OVERLAY_MA, band=OVERLAY_BAND)
    years = sorted({pd.Timestamp(d).year for d in dts.unique()})

    def run(stop, overlay):
        d = run_backtest(scores, fwd, FixedPeriod(top_k=TOP_K, period=PERIOD),
                         cost_model(PROFILE), capital=CAPITAL,
                         exposure=(exposure if overlay else None), stop_loss=stop)["daily"]
        m = net_metrics(d); m["calmar"] = _calmar(m)
        yr = {y: net_metrics(d.loc[pd.to_datetime(d.index).year == y])["net_cagr"] for y in years}
        m["neg_years"] = sum(1 for y in years if np.isfinite(yr[y]) and yr[y] < 0)
        m["y2022"], m["y2023"] = yr.get(2022), yr.get(2023)
        return m

    out = {}
    for overlay in (False, True):
        tag = "+overlay" if overlay else "no-overlay"
        hdr = (f"{'stop_loss':>9} {'net_cagr':>9} {'net_ir':>7} {'max_dd':>8} {'Calmar':>7} "
               f"{'turnov':>7} {'win':>6} {'negY':>5} {'2022':>7} {'2023':>7}")
        print(f"\n=== factor-2model — stop-loss sweep [{tag}] (fixed/5d, ¥100k, small) ===")
        print(hdr); print("-" * len(hdr))
        for stop in STOPS:
            m = run(stop, overlay)
            out[f"{'none' if stop is None else stop}|{tag}"] = m
            print(f"{('none' if stop is None else f'{stop:.0%}'):>9} {_pct(m['net_cagr']):>9} "
                  f"{_num(m['net_ir']):>7} {_pct(m['max_drawdown']):>8} {_num(m['calmar']):>7} "
                  f"{_num(m['avg_turnover'],3):>7} {_pct(m['win_rate']):>6} {m['neg_years']:>5} "
                  f"{_pct(m['y2022']):>7} {_pct(m['y2023']):>7}")

    Path("logs/eval_stops_summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("\nwrote logs/eval_stops_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
