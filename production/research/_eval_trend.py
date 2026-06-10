# production/_eval_trend.py
"""Evaluate the trend-aware ranking gate on the long-window OOF predictions and
run a live falling-knife sanity check.

Run from the repo root (NEVER `python file.py`):

  F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._eval_trend \
      > logs/eval_trend.log 2>&1

What it does
------------
1. Loads production/reports/oof_2model_2021_2026.pkl and extracts the `score`
   series (MultiIndex datetime, instrument; higher = better).
2. Initializes qlib, then loads TRAILING trend features for the OOF instruments
   over the OOF date span in ONE QlibDataLoader call (restricted to OOF
   instruments to stay light).
3. Loads forward returns ONCE via production.backtest.data.load_fwd_returns.
4. For each variant -- (a) baseline no filter, (b) soft close>MA20, (c) strict
   -- runs production.backtest.run.build_report, both WITHOUT and WITH the P3
   market-regime overlay (compute_exposure on a synthetic market proxy).
5. Prints a comparison table: net_cagr, net_ir, avg_turnover, max_drawdown,
   Calmar (= net_cagr / |max_drawdown|), plus per-regime net_cagr.
6. Live sanity check: latest cross-section of examples/mlruns/pred_2026-06-02.pkl,
   top-15 raw vs top-15 after the soft gate, and how many raw top-15 are
   falling knives (close < MA20).

This script does NOT retrain anything and does NOT modify engine internals.
"""
from __future__ import annotations

# --- sys.path fixup (installed qlib ahead of ./qlib source) ---
import sys as _sys
import sysconfig as _sysconfig

_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)
try:  # avoid GBK errors on ¥ / Chinese output
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
# --- end fixup ---

from pathlib import Path

import numpy as np
import pandas as pd

from production.trend_filter import (
    apply_trend_filter,
    TREND_FEATURE_EXPRS,
    TREND_FEATURE_NAMES,
    is_downtrend,
)

OOF_PKL = "production/reports/oof_2model_2021_2026.pkl"
LIVE_PKL = "examples/mlruns/pred_2026-06-02.pkl"
CONFIG = "production/configs/rolling_ensemble.yaml"

TOP_K = 5
EXIT_K = 10
PERIOD = 5
CAPITAL = 100_000.0
PROFILE = "small"

# overlay params (match the task spec / P3 design)
OVERLAY_MA_WINDOW = 60
OVERLAY_BAND = 0.10


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _extract_score(pred, score_col: str = "score") -> pd.Series:
    if isinstance(pred, pd.DataFrame):
        col = score_col if score_col in pred.columns else pred.columns[0]
        s = pred[col]
    else:
        s = pred
    s = s.copy()
    s.index = s.index.set_names(["datetime", "instrument"])
    return s.rename("score").dropna()


def _load_trend_features_via_qlib(instruments, start, end) -> pd.DataFrame:
    """One QlibDataLoader call -> trailing trend-feature frame indexed by
    (datetime, instrument) with columns TREND_FEATURE_NAMES.

    We load the raw expressions (close, MAs, momentum) and assemble the
    boolean flags here so the flag logic is identical to the unit-tested
    compute_trend_features (NaN during warm-up -> conservative fail)."""
    from qlib.data.dataset.loader import QlibDataLoader
    from production.backtest.data import init_qlib_from_config

    init_qlib_from_config(CONFIG)
    raw_names = ["close", "ma5", "ma10", "ma20", "ma60", "momentum"]
    loader = QlibDataLoader(config={"feature": (TREND_FEATURE_EXPRS, raw_names)})
    df = loader.load(instruments=instruments, start_time=start, end_time=end)
    # QlibDataLoader returns MultiIndex columns ('feature', name); flatten to name.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    if df.index.names[0] == "instrument":
        df = df.swaplevel().sort_index()
    df.index = df.index.set_names(["datetime", "instrument"])
    df = df.sort_index()

    close, ma5, ma10, ma20, ma60, mom = (
        df["close"], df["ma5"], df["ma10"], df["ma20"], df["ma60"], df["momentum"])

    def _flag(cond, valid):
        return cond.astype(float).where(valid, np.nan)

    out = pd.DataFrame(index=df.index)
    out["close_gt_ma20"] = _flag(close > ma20, ma20.notna())
    out["close_gt_ma60"] = _flag(close > ma60, ma60.notna())
    aligned = (ma5 > ma10) & (ma10 > ma20)
    out["ma_aligned"] = _flag(aligned, ma5.notna() & ma10.notna() & ma20.notna())
    out["momentum"] = mom
    out["ma20"] = ma20
    out["ma60"] = ma60
    return out[TREND_FEATURE_NAMES].sort_index()


def _calmar(metrics: dict) -> float:
    dd = metrics.get("max_drawdown", float("nan"))
    cagr = metrics.get("net_cagr", float("nan"))
    if dd is None or not np.isfinite(dd) or abs(dd) < 1e-12:
        return float("nan")
    return cagr / abs(dd)


def _date_coverage(scores: pd.Series, fwd: pd.Series) -> int:
    """Number of dates the engine will actually simulate (scores ∩ fwd)."""
    fwd_dates = set(fwd.index.get_level_values("datetime").unique())
    return sum(1 for d in scores.index.get_level_values("datetime").unique() if d in fwd_dates)


def _run_variant(name, scores, fwd, exposure):
    from production.backtest.run import build_report
    # CANONICAL config = fixed/hold-5/5-day (the +19% baseline in
    # 2026-06-01-longwindow-lgbm-results.md). NOT banded — banded underperforms
    # fixed on this universe (2026-05-31-net-return-results.md).
    rep = build_report(scores, fwd, policy_name="fixed", top_k=TOP_K, period=PERIOD,
                       exit_k=EXIT_K, capital=CAPITAL, profile=PROFILE, exposure=exposure)
    m = rep["metrics"]
    return {
        "variant": name,
        "net_cagr": m["net_cagr"],
        "net_ir": m["net_ir"],
        "avg_turnover": m["avg_turnover"],
        "max_drawdown": m["max_drawdown"],
        "calmar": _calmar(m),
        "win_rate": m["win_rate"],
        "n_days": m["n_days"],
        "final_nav": rep["final_nav"],
        "regimes": rep["regimes"],
        "n_dates_scored": _date_coverage(scores, fwd),
    }


def _fmt_pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+7.2%}"


def _fmt_num(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _print_table(rows):
    hdr = (f"{'variant':<26} {'net_cagr':>9} {'net_ir':>7} {'turnover':>9} "
           f"{'max_dd':>9} {'Calmar':>7} {'win':>6} {'days':>5} {'scored':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['variant']:<26} {_fmt_pct(r['net_cagr']):>9} {_fmt_num(r['net_ir']):>7} "
              f"{_fmt_num(r['avg_turnover'], 3):>9} {_fmt_pct(r['max_drawdown']):>9} "
              f"{_fmt_num(r['calmar']):>7} {_fmt_pct(r['win_rate']):>6} "
              f"{r['n_days']:>5} {r['n_dates_scored']:>7}")


def _print_regime_table(rows):
    # union of regime keys, in chronological order
    keys = []
    for r in rows:
        for k in r["regimes"]:
            if k not in keys:
                keys.append(k)
    keys.sort()
    print("\nPer-regime net_cagr:")
    hdr = f"{'variant':<26} " + " ".join(f"{k.split('__')[0][:7]:>9}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = []
        for k in keys:
            seg = r["regimes"].get(k, {})
            cells.append(_fmt_pct(seg.get("net_cagr")))
        print(f"{r['variant']:<26} " + " ".join(f"{c:>9}" for c in cells))


# --------------------------------------------------------------------------- #
# main eval                                                                    #
# --------------------------------------------------------------------------- #
def run_oof_eval() -> list[dict]:
    print("=" * 78)
    print("TREND-AWARE RANKING — long-window OOF evaluation")
    print("=" * 78)
    pred = pd.read_pickle(OOF_PKL)
    scores = _extract_score(pred)
    instruments = sorted(scores.index.get_level_values("instrument").unique())
    dates = scores.index.get_level_values("datetime")
    start, end = str(dates.min().date()), str(dates.max().date())
    print(f"OOF: {len(scores):,} rows | {len(instruments)} instruments | "
          f"{dates.nunique()} dates | {start} .. {end}")

    print("loading trailing trend features via qlib (one call) ...")
    feats = _load_trend_features_via_qlib(instruments, start, end)
    print(f"trend features: {len(feats):,} rows, cols={list(feats.columns)}")

    print("loading forward returns once ...")
    from production.backtest.data import load_fwd_returns
    fwd = load_fwd_returns(instruments, start, end, config_path=CONFIG)
    print(f"fwd returns: {len(fwd):,} rows")

    # build the P3 overlay once (shared across the +overlay variants)
    print("building P3 market-regime overlay ...")
    from production.backtest.market import load_market_proxy
    from production.backtest.regime import compute_exposure
    mkt = load_market_proxy(instruments, start, end, config_path=CONFIG)
    exposure = compute_exposure(mkt, method="trend_ma",
                                ma_window=OVERLAY_MA_WINDOW, band=OVERLAY_BAND)
    exposure = exposure.rename("trend_ma60")

    gated = {
        "baseline (no gate)": apply_trend_filter(scores, feats, mode="none"),
        "soft (close>MA20)": apply_trend_filter(scores, feats, mode="soft"),
        "strict (align+mom)": apply_trend_filter(scores, feats, mode="strict"),
    }
    for k, g in gated.items():
        full = scores.index.get_level_values("datetime").nunique()
        keptd = g.index.get_level_values("datetime").nunique()
        print(f"  gate {k:<22}: kept {len(g):,}/{len(scores):,} rows "
              f"({len(g)/max(1,len(scores)):.1%}); dates {keptd}/{full}")

    rows = []
    for label, g in gated.items():
        rows.append(_run_variant(label, g, fwd, exposure=None))
    for label, g in gated.items():
        rows.append(_run_variant(label + " +overlay", g, fwd, exposure=exposure))

    print("\n" + "=" * 78)
    print("COMPARISON TABLE  (policy=fixed top_k=5 period=5 "
          "capital=100000 profile=small) [canonical hold-5/5-day]")
    print("=" * 78)
    _print_table(rows)
    _print_regime_table(rows)
    return rows


# --------------------------------------------------------------------------- #
# live falling-knife sanity check                                              #
# --------------------------------------------------------------------------- #
def run_live_sanity(top_n: int = 15) -> None:
    print("\n" + "=" * 78)
    print(f"LIVE SANITY CHECK — {LIVE_PKL} (top-{top_n} raw vs soft-gated)")
    print("=" * 78)
    pred = pd.read_pickle(LIVE_PKL)
    latest = pred.index.get_level_values("datetime").max()
    cross = pred.xs(latest, level="datetime")
    score = cross["score"] if "score" in cross.columns else cross.iloc[:, 0]
    insts = sorted(pred.index.get_level_values("instrument").unique())

    # close history for MA20 (>=80 sessions for a clean MA20 at `latest`)
    from qlib.data.dataset.loader import QlibDataLoader
    from production.backtest.data import init_qlib_from_config
    init_qlib_from_config(CONFIG)
    start = str((latest - pd.Timedelta(days=160)).date())
    end = str(latest.date())
    px = QlibDataLoader(config={"feature": (["$close"], ["close"])}).load(
        instruments=insts, start_time=start, end_time=end)
    s = px.iloc[:, 0] if isinstance(px, pd.DataFrame) else px
    if s.index.names[0] == "instrument":
        s = s.swaplevel().sort_index()
    s.index = s.index.set_names(["datetime", "instrument"])

    def _closevec(inst):
        return s.xs(inst, level="instrument").sort_index().values.astype(float)

    raw_top = list(score.sort_values(ascending=False).head(top_n).index)
    n_knife = sum(1 for inst in raw_top if is_downtrend(_closevec(inst)))

    # build a single-date feature frame + apply soft gate
    feats_rows = {}
    for inst in insts:
        cl = _closevec(inst)
        if cl.size >= 20 and np.isfinite(cl[-1]):
            ma20 = cl[-20:].mean()
            feats_rows[inst] = float(cl[-1] > ma20)
    feat = pd.DataFrame({
        "close_gt_ma20": pd.Series(feats_rows),
    })
    # fill the other marker columns so _is_feature_frame recognises it
    feat["ma_aligned"] = np.nan
    feat["momentum"] = np.nan
    feat.index = pd.MultiIndex.from_product([[latest], feat.index],
                                            names=["datetime", "instrument"])
    score_mi = score.copy()
    score_mi.index = pd.MultiIndex.from_product([[latest], score_mi.index],
                                                names=["datetime", "instrument"])
    soft = apply_trend_filter(score_mi, feat, mode="soft")
    soft_top = list(soft.sort_values(ascending=False).head(top_n)
                    .index.get_level_values("instrument"))

    print(f"latest cross-section: {latest.date()} | {len(insts)} instruments")
    print(f"\nRAW top-{top_n}  (falling knives flagged ✗):")
    for i, inst in enumerate(raw_top, 1):
        cl = _closevec(inst)
        knife = is_downtrend(cl)
        ma20 = cl[-20:].mean() if cl.size >= 20 else float("nan")
        vs = cl[-1] / ma20 - 1 if ma20 == ma20 else float("nan")
        print(f"  {i:>2} {inst:<10} score={score[inst]:+.4f}  vsMA20={vs:+6.1%}  "
              f"{'✗ FALLING-KNIFE' if knife else 'ok'}")
    print(f"\n  -> {n_knife}/{top_n} of RAW top-{top_n} are falling knives "
          f"(close<MA20) = {n_knife/top_n:.0%}")

    print(f"\nSOFT-GATED top-{top_n} (close>MA20 enforced):")
    for i, inst in enumerate(soft_top, 1):
        cl = _closevec(inst)
        ma20 = cl[-20:].mean() if cl.size >= 20 else float("nan")
        vs = cl[-1] / ma20 - 1 if ma20 == ma20 else float("nan")
        print(f"  {i:>2} {inst:<10} score={score[inst]:+.4f}  vsMA20={vs:+6.1%}")
    knives_after = sum(1 for inst in soft_top if is_downtrend(_closevec(inst)))
    print(f"\n  -> {knives_after}/{len(soft_top)} of SOFT-GATED top are falling knives "
          f"(should be 0 by construction)")
    print(f"  -> falling-knife reduction: {n_knife} -> {knives_after} in the top-{top_n}")


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    rows = run_oof_eval()
    try:
        run_live_sanity()
    except Exception as e:  # live check is best-effort; OOF eval is the deliverable
        print(f"\n[live sanity check skipped: {type(e).__name__}: {e}]")
    # machine-readable dump for the spec
    import json
    slim = [{k: v for k, v in r.items() if k != "regimes"} for r in rows]
    Path("logs/eval_trend_summary.json").write_text(
        json.dumps(slim, indent=2, default=float), encoding="utf-8")
    print("\nwrote logs/eval_trend_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
