# production/research/_eval_etf_real.py
"""A2a: validate the ETF path on REAL tradeable ETFs/indices (not the synthetic
equal-weight proxy). Fetch daily front-adjusted close via baostock over the OOF span,
compute buy&hold + MA60-timed net metrics with the etf cost profile, compare to the
synthetic proxy (+14.4% / maxDD -26%). Run from worktree (needs OOF for span +
baostock network). ETFs may lack baostock coverage -> the underlying indices are
included as a fallback (index = price-only, ~2-3%/yr below a total-return ETF).

Run: F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._eval_etf_real \
  > logs/eval_etf_real.log 2>&1
"""
from production.research._harness import bootstrap, OOF_2MODEL
bootstrap()

import json
from pathlib import Path
import pandas as pd
CAPITAL = 10_000.0
MA_WINDOW, BAND = 60, 0.10
# broad beta the user could actually buy at ¥10k (ETF) + underlying index fallback
ETFS = [
    ("沪深300ETF 510300", "sh.510300"),
    ("中证500ETF 510500", "sh.510500"),
    ("上证50ETF 510050", "sh.510050"),
    ("创业板ETF 159915", "sz.159915"),
    ("沪深300指数 000300", "sh.000300"),
    ("中证500指数 000905", "sh.000905"),
    ("中证800指数 000906", "sh.000906"),
]
CACHE = Path("production/reports")


def fetch_nav(code: str, start: str, end: str) -> pd.Series:
    """Daily front-adjusted close (return-faithful) for an ETF/index via baostock,
    cached to parquet. Returns Series datetime->close (empty on no-data/connection fail)."""
    fp = CACHE / f"etf_nav_{code.replace('.', '_')}_{start}_{end}.parquet"
    if fp.exists():
        s = pd.read_parquet(fp)["close"]
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    from production.intraday.fetch_5min import _query
    d = _query(lambda bs: bs.query_history_k_data_plus(
        code, "date,close", start_date=start, end_date=end, frequency="d", adjustflag="2"))
    if d is None or not len(d):
        return pd.Series(dtype=float)
    close = pd.to_numeric(d["close"], errors="coerce")
    s = pd.Series(close.values, index=pd.to_datetime(d["date"].values)).dropna().sort_index()
    s = s[s > 0]
    if len(s):
        CACHE.mkdir(parents=True, exist_ok=True)
        s.rename("close").to_frame().to_parquet(fp)
    return s


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    from production.score_utils import score_of
    from production.backtest.regime import compute_exposure
    from production.backtest.etf_timing import simulate_etf_timing
    from production.backtest.costs import cost_model
    from production.backtest.metrics_net import net_metrics, tail_stats

    two = pd.read_pickle(OOF_2MODEL)
    dts = score_of(two).index.get_level_values("datetime")
    start, end = str(dts.min().date()), str(dts.max().date())
    print(f"span {start}..{end}, capital Y{CAPITAL:,.0f}, etf cost profile\n")
    print(f"{'instrument':>20} {'mode':>8} {'net_cagr':>9} {'maxDD':>8} {'IR':>6} {'neg%':>6}")
    print("-" * 62)
    out = {}
    for label, code in ETFS:
        nav = fetch_nav(code, start, end)
        if nav.empty or len(nav) < MA_WINDOW + 20:
            print(f"{label:>20} {'NO-DATA':>8}  (got {len(nav)} bars)")
            out[code] = {"label": label, "error": "no-data", "n": int(len(nav))}
            continue
        ret = nav.pct_change().dropna()
        exposure = compute_exposure(nav, method="trend_ma", ma_window=MA_WINDOW, band=BAND)
        bh = simulate_etf_timing(ret, pd.Series(1.0, index=ret.index), cost_model("etf"), capital=CAPITAL)
        tm = simulate_etf_timing(ret, exposure, cost_model("etf"), capital=CAPITAL)
        rec = {"label": label}
        for mode, led in [("buyhold", bh), ("timed", tm)]:
            m = net_metrics(led); ts = tail_stats(led["net"])
            rec[mode] = {"metrics": m, "tail": ts}
            print(f"{label:>20} {mode:>8} {m['net_cagr']:>+9.2%} {m['max_drawdown']:>+8.2%} "
                  f"{m['net_ir']:>6.2f} {ts['neg_period_pct']:>6.1%}")
        out[code] = rec
    Path("logs/eval_etf_real.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("\nsynthetic proxy baseline: buyhold +14.36% / maxDD -26.45% / IR 0.79")
    print("(index rows are price-only; a real total-return ETF runs ~2-3%/yr higher)")
    print("wrote logs/eval_etf_real.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
