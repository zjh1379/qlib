# production/research/_reconcile_live.py
"""Decompose each real-money trade into SIGNAL (did the model rank it top-k? would the
backtest's open->open have profited?) vs EXECUTION (did you pay above the backtest's
entry open = chase? was it a 涨停 gap = backtest open fill unattainable?). Seeds the
forward-test journal.

Run: F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._reconcile_live \
  --trades production/reports/live_trades.csv > logs/reconcile_live.log 2>&1
"""
import sys as _sys, sysconfig as _sysconfig
_P = _sysconfig.get_paths().get("purelib")
if _P and _P not in _sys.path[:1]:
    _sys.path.insert(0, _P)
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import json
from pathlib import Path
import pandas as pd

OOF_FAC = "production/reports/oof_lgbmfac_2021_2026.pkl"
OOF_2MODEL = "production/reports/oof_2model_2021_2026.pkl"
CONFIG = "production/configs/rolling_ensemble.yaml"


def decompose_trade(fill_price, backtest_open, bt_fwd_ret, rank, buyable, top_k=5) -> dict:
    """entry_premium = fill/open - 1 (>0 => chased above the open the backtest assumes).
    in_topk = was it a model top-k pick (signal). bt_fwd_ret = backtest open->open 1d
    return on the decision day (signal-quality proxy). buyable = open below 涨停 cap
    (False => backtest's open fill was unattainable)."""
    has_open = backtest_open is not None and backtest_open == backtest_open and backtest_open != 0
    return {
        "rank": rank,
        "in_topk": (rank is not None and rank <= top_k),
        "bt_fwd_ret": bt_fwd_ret,
        "entry_premium_pct": (fill_price / backtest_open - 1) if has_open else float("nan"),
        "buyable_at_open": (None if buyable is None else bool(buyable)),
    }


def _rank_on(scores: pd.Series, date, inst) -> "int | None":
    try:
        cross = scores.xs(date, level="datetime").dropna().sort_values(ascending=False)
    except KeyError:
        return None
    order = list(cross.index)
    return order.index(inst) + 1 if inst in order else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="production/reports/live_trades.csv")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="logs/reconcile_live.json")
    args = ap.parse_args()

    tr = pd.read_csv(args.trades, dtype={"instrument": str})
    tr["trade_date"] = pd.to_datetime(tr["trade_date"])

    from production.score_utils import rebuild_2model
    from production.backtest.data import load_fwd_returns
    from production.backtest.executability import load_entry_ohlc, buyable_mask

    scores = rebuild_2model(pd.read_pickle(OOF_FAC), pd.read_pickle(OOF_2MODEL))
    insts = sorted(tr["instrument"].unique())
    start = str(tr["trade_date"].min().date())
    end = str((tr["trade_date"].max() + pd.Timedelta(days=20)).date())
    fwd = load_fwd_returns(insts, start, end, config_path=CONFIG)
    ohlc = load_entry_ohlc(insts, start, end, config_path=CONFIG)
    buyable = buyable_mask(ohlc)

    rows = []
    for _, t in tr.iterrows():
        d, inst = t["trade_date"], t["instrument"]
        bo = ohlc["entry_open"].get((d, inst), float("nan"))
        dec = decompose_trade(
            fill_price=float(t["fill_price"]),
            backtest_open=float(bo),
            bt_fwd_ret=float(fwd.get((d, inst), float("nan"))),
            rank=_rank_on(scores, d, inst),
            buyable=(bool(buyable.get((d, inst))) if (d, inst) in buyable.index else None),
            top_k=args.top_k)
        dec.update({"trade_date": str(d.date()), "instrument": inst,
                    "fill_price": float(t["fill_price"]),
                    "entry_timing": t.get("entry_timing", ""), "note": t.get("note", "")})
        rows.append(dec)
        print(f"{inst} {str(d.date())} [{dec.get('note','')}]: rank={dec['rank']} "
              f"in_top{args.top_k}={dec['in_topk']} buyable={dec['buyable_at_open']} "
              f"bt_fwd_1d={dec['bt_fwd_ret']:+.2%} entry_premium={dec['entry_premium_pct']:+.2%}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
