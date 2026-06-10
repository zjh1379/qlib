"""Does LambdaRank beat MSE? Compares the LambdaRank LGBM (objective=lambdarank,
same factor features) against the MSE factor champion (+31%) and the original
baseline, under the canonical fixed/hold-5/5d net-of-cost backtest. Reports
full + ex-2020 metrics, per-year, and paired daily-net t-tests.

Run AFTER the rank backfill + pooling:
  python -m production.research._pool_rank
  python -X utf8 -m production.research._eval_rank > logs/eval_rank.log 2>&1
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

OOF_RANK = "production/reports/oof_lgbmrank_2021_2026.pkl"   # lambdarank LGBM
OOF_FAC = "production/reports/oof_lgbmfac_2021_2026.pkl"     # mse factor LGBM (champion half)
OOF_2MODEL = "production/reports/oof_2model_2021_2026.pkl"   # for ALSTM cols + baseline
CONFIG = "production/configs/rolling_ensemble.yaml"
TOP_K, PERIOD, CAPITAL, PROFILE = 5, 5, 100_000.0, "small"
EX2020 = "2021-01-01"


def _pct(x): return "n/a" if x is None or not np.isfinite(x) else f"{x:+7.2%}"
def _num(x, n=2): return "n/a" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    for p in (OOF_RANK, OOF_FAC, OOF_2MODEL):
        if not Path(p).exists():
            print(f"MISSING {p} — run the rank backfill + _pool_rank first."); return 1

    from production.score_utils import score_of as _score_of, rebuild_2model as _rebuild_2model
    from production.research._eval_robustness import _paired_t, _calmar
    from production.backtest.metrics_net import net_metrics
    from production.backtest.engine import run_backtest
    from production.backtest.rebalance import FixedPeriod
    from production.backtest.costs import cost_model
    from production.backtest.data import load_fwd_returns

    rank = pd.read_pickle(OOF_RANK)
    fac = pd.read_pickle(OOF_FAC)
    two = pd.read_pickle(OOF_2MODEL)
    scores = {
        "baseline-2model": _score_of(two),
        "factor-2model(mse,+31%)": _rebuild_2model(fac, two),
        "rank-2model(lambdarank)": _rebuild_2model(rank, two),
        "factor-LGBM(mse)": _score_of(fac),
        "rank-LGBM(lambdarank)": _score_of(rank),
    }
    insts = sorted(set().union(*[set(s.index.get_level_values("instrument")) for s in scores.values()]))
    dts = pd.DatetimeIndex(sorted(set().union(*[set(s.index.get_level_values("datetime")) for s in scores.values()])))
    fwd = load_fwd_returns(insts, str(dts.min().date()), str(dts.max().date()), config_path=CONFIG)

    def bt(s):
        return run_backtest(s, fwd, FixedPeriod(top_k=TOP_K, period=PERIOD), cost_model(PROFILE), capital=CAPITAL)["daily"]
    daily = {k: bt(s) for k, s in scores.items()}
    years = sorted({pd.Timestamp(d).year for d in daily["rank-2model(lambdarank)"].index})

    def metrics(d, sd=None):
        if sd is not None:
            d = d.loc[pd.to_datetime(d.index) >= pd.Timestamp(sd)]
        m = net_metrics(d); m["calmar"] = _calmar(m); return m

    hdr = f"{'model':<26} {'net_cagr':>9} {'net_ir':>7} {'max_dd':>8} {'Calmar':>7} {'exC_cagr':>9} {'negY':>5}"
    print("LAMBDARANK vs MSE (fixed/hold-5/5d, ¥100k, small cost)")
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    rows = {}
    for k, d in daily.items():
        m = metrics(d); mx = metrics(d, EX2020)
        negY = sum(1 for y in years if (lambda v: np.isfinite(v) and v < 0)(net_metrics(d.loc[pd.to_datetime(d.index).year == y])["net_cagr"]))
        rows[k] = {"full": m, "ex2020_cagr": mx["net_cagr"], "negY": negY}
        print(f"{k:<26} {_pct(m['net_cagr']):>9} {_num(m['net_ir']):>7} {_pct(m['max_drawdown']):>8} "
              f"{_num(m['calmar']):>7} {_pct(mx['net_cagr']):>9} {negY:>5}")

    print("\nPER-YEAR net_cagr — factor(mse) vs rank(lambdarank) 2-model:")
    yh = f"{'model':<26} " + " ".join(f"{y:>8}" for y in years); print(yh); print("-" * len(yh))
    for k in ("factor-2model(mse,+31%)", "rank-2model(lambdarank)"):
        cells = [_pct(net_metrics(daily[k].loc[pd.to_datetime(daily[k].index).year == y])["net_cagr"]) for y in years]
        print(f"{k:<26} " + " ".join(f"{c:>8}" for c in cells))

    print("\nPAIRED t-TEST (daily net, treatment vs control):")
    tests = {
        "rank-2model vs factor-2model (lambdarank gain over mse)": ("rank-2model(lambdarank)", "factor-2model(mse,+31%)"),
        "rank-2model vs baseline-2model": ("rank-2model(lambdarank)", "baseline-2model"),
        "rank-LGBM vs factor-LGBM": ("rank-LGBM(lambdarank)", "factor-LGBM(mse)"),
    }
    tt = {}
    for label, (a, b) in tests.items():
        r = _paired_t(daily[a]["net"], daily[b]["net"]); tt[label] = r
        sig = "***" if r["p"] < 0.01 else "**" if r["p"] < 0.05 else "*" if r["p"] < 0.10 else "ns"
        print(f"  {label:<52} mean~{r.get('mean_annual', float('nan')):+.1%}/yr  t={_num(r['t'])} p={_num(r['p'],4)} [{sig}]")

    Path("logs/eval_rank_summary.json").write_text(json.dumps({"rows": rows, "ttests": tt}, indent=2, default=float), encoding="utf-8")
    print("\nwrote logs/eval_rank_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
