# production/research/_eval_user_exec.py
"""Model the user's ACTUAL exit on the deployable top-3: SELL in the AFTERNOON a few
days later (= daily CLOSE at d+HOLD) vs the backtest baseline (next-OPEN at d+HOLD),
hold ~a few days (sweep). Entry = open(d+1) as the optimistic proxy — P1 showed the
user's first-30-min intraday entry costs ~10-16pp vs open (picks bounce from the open),
so open is best-case and the real entry is somewhat worse. Daily data only (fast).

Run: F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._eval_user_exec \
  > logs/eval_user_exec.log 2>&1
"""
from production.research._harness import bootstrap, OOF_FAC, OOF_2MODEL, CONFIG, champion_scores
bootstrap()

import json
from pathlib import Path
import numpy as np
import pandas as pd
TOP_K = 3
HOLDS = [2, 3, 4, 5]
COST_BPS = 30.0  # round-trip per trade, approx small-capital (万2.5x2 + 印花 + 滑点)


def _daily_oc(insts, start, end) -> pd.DataFrame:
    """Adjusted daily open+close per (datetime,instrument). Force column names by
    position (QlibDataLoader returns MultiIndex/expr-named cols)."""
    from production.qlib_features import load_features
    return load_features(insts, start, end, ["$open", "$close"], ["open", "close"], config_path=CONFIG)


def _metrics(per_period: pd.Series, hold: int) -> dict:
    eq = (1 + per_period).cumprod()
    n = len(per_period)
    ann = (eq.iloc[-1] ** (252 / (hold * n)) - 1) if n and eq.iloc[-1] > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1).min()) if n else float("nan")
    return {"net_cagr": ann, "calmar": (ann / abs(dd)) if dd and abs(dd) > 1e-9 else float("nan"),
            "max_dd": dd, "win": float((per_period > 0).mean()) if n else float("nan"),
            "n_periods": n}


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    from production.intraday.exec_backtest import enumerate_trades
    scores = champion_scores()
    insts = sorted(scores.index.get_level_values("instrument").unique())
    dts = scores.index.get_level_values("datetime")
    start, end = str(dts.min().date()), str(dts.max().date())
    oc = _daily_oc(insts, start, end)
    opens, closes = oc["open"], oc["close"]

    out = {}
    print(f"top-{TOP_K}, entry=open(d+1) (P1: real first-30min entry ~10-16pp worse), {COST_BPS}bp/rt")
    print(f"{'hold':>4} {'exit':>6} {'net_cagr':>9} {'maxDD':>8} {'Calmar':>7} {'win':>6} {'n':>5}")
    print("-" * 50)
    for hold in HOLDS:
        trades = enumerate_trades(scores, top_k=TOP_K, period=hold)
        for exit_field, exit_src in [("open", opens), ("close", closes)]:
            per: dict = {}
            for t in trades:
                eo = opens.get((t["entry_date"], t["instrument"]))
                xo = exit_src.get((t["exit_date"], t["instrument"]))
                if eo is None or xo is None or not (eo > 0):
                    continue
                per.setdefault(t["decision_date"], []).append(float(xo) / float(eo) - 1 - COST_BPS / 1e4)
            ser = pd.Series({d: float(np.mean(v)) for d, v in per.items()}).sort_index()
            m = _metrics(ser, hold)
            out[f"hold{hold}_{exit_field}"] = m
            print(f"{hold:>4} {exit_field:>6} {m['net_cagr']:>+9.2%} {m['max_dd']:>+8.2%} "
                  f"{m['calmar']:>7.2f} {m['win']:>6.1%} {m['n_periods']:>5}")
    Path("logs/eval_user_exec.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("\nwrote logs/eval_user_exec.json  (close=your afternoon exit; open=backtest baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
