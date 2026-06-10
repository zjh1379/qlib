"""Throwaway P3-T5 driver: sweep regime-overlay params on the 2-model OOF,
pick the config hitting DD in [-30%,-20%] with max net CAGR, write result JSON.
Run: F:/Tools/Anaconda/envs/qlib/python.exe production/_t5_overlay_sweep.py
(Real .py + __main__ guard so qlib's multiprocessing workers can re-import safely.)"""
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    from production.backtest.run import extract_score_series
    from production.backtest.data import load_fwd_returns
    from production.backtest.market import load_market_proxy
    from production.backtest.regime import compute_exposure
    from production.backtest.engine import run_backtest
    from production.backtest.rebalance import FixedPeriod
    from production.backtest.costs import cost_model
    from production.backtest.metrics_net import net_metrics, net_regime

    pred = pd.read_pickle("production/reports/oof_2model_2021_2026.pkl")
    scores = extract_score_series(pred, "score").dropna()
    inst = sorted(scores.index.get_level_values("instrument").unique())
    dts = scores.index.get_level_values("datetime")
    start, end = str(dts.min().date()), str(dts.max().date())
    fwd = load_fwd_returns(inst, start, end)
    mkt = load_market_proxy(inst, start, end)
    cm = cost_model("small")
    pol = FixedPeriod(top_k=5, period=5)
    cap = 100_000.0

    def ev(e):
        d = run_backtest(scores, fwd, pol, cm, capital=cap, exposure=e)["daily"]
        return net_metrics(d), d

    bm, _ = ev(None)
    print(f"BASELINE no-overlay: cagr={bm['net_cagr']:+.3f} dd={bm['max_drawdown']:+.3f} "
          f"ir={bm['net_ir']:+.3f} calmar={bm['net_cagr']/abs(bm['max_drawdown']):.2f}", flush=True)

    rows = []
    for maw in (60, 120, 200):
        for mine in (0.0, 0.2):
            for vt in (None, 0.15):
                e = compute_exposure(mkt, ma_window=maw, band=0.10, min_exposure=mine, vol_target=vt)
                m, _ = ev(e)
                cal = m["net_cagr"] / abs(m["max_drawdown"]) if m["max_drawdown"] < 0 else float("nan")
                rows.append({"ma": maw, "min": mine, "vt": vt, **m, "calmar": cal})
                print(f"ma{maw} min{mine} vt{str(vt):>4}: cagr={m['net_cagr']:+.3f} "
                      f"dd={m['max_drawdown']:+.3f} ir={m['net_ir']:+.3f} calmar={cal:.2f}", flush=True)

    cand = [r for r in rows if -0.30 <= r["max_drawdown"] <= -0.20]
    best = (max(cand, key=lambda r: r["net_cagr"]) if cand
            else max(rows, key=lambda r: (r["calmar"] if r["calmar"] == r["calmar"] else -9)))
    print("PICK:", {k: best[k] for k in ("ma", "min", "vt", "net_cagr", "max_drawdown", "net_ir", "calmar")}, flush=True)

    e = compute_exposure(mkt, ma_window=best["ma"], band=0.10, min_exposure=best["min"], vol_target=best["vt"])
    m, d = ev(e)
    regimes = net_regime(d)
    out = {"baseline_metrics": bm, "pick_params": {k: best[k] for k in ("ma", "min", "vt")},
           "pick_metrics": m, "pick_regimes": regimes}
    Path("production/reports/risk_overlay_result.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("REGIMES net_ir:", {k.split('__')[0]: round(v["net_ir"], 2) for k, v in regimes.items()}, flush=True)
    print("wrote production/reports/risk_overlay_result.json", flush=True)


if __name__ == "__main__":
    main()
