# production/research/forward_journal.py
"""Forward paper-trade journal for the deployable top-K strategy (A2 result:
single-stock top-3, fixed/5d). Accumulates REAL out-of-sample evidence:

  snapshot : log the SERVED top-K picks for the latest as-of date (append-only)
  reconcile: fill realized open(d+1)->open(d+1+HOLD) returns + 涨停 buyability for
             picks whose holding period has elapsed, and print forward cumulative.

Runs from the MAIN repo (E:/Projects/qlib), where examples/mlruns + the served
recorder live — same as daily_inference. (From a worktree examples/mlruns is empty,
so `snapshot` will report no served recorder.)

  python -X utf8 -m production.research.forward_journal snapshot --top-k 3
  python -X utf8 -m production.research.forward_journal reconcile
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

REPO_ROOT = Path(__file__).resolve().parents[2]
JOURNAL = REPO_ROOT / "production" / "reports" / "forward_journal.csv"
CONFIG = "production/configs/rolling_ensemble.yaml"
EXPERIMENT = "rolling_v2_ensemble"
HOLD = 5  # trading-day holding period, matches the deployable fixed/5 config
COLS = ["as_of_date", "rank", "instrument", "name", "score",
        "entry_open", "exit_open", "realized_ret", "buyable_at_open", "reconciled"]


def select_topk(pred, as_of, k: int) -> pd.DataFrame:
    """Top-k by score on `as_of` date. Returns DataFrame [rank, instrument, score].
    Pure — unit-tested."""
    s = pred["score"] if isinstance(pred, pd.DataFrame) else pred
    cross = s.dropna().xs(as_of, level="datetime").sort_values(ascending=False)
    top = cross.head(k)
    return pd.DataFrame({"rank": list(range(1, len(top) + 1)),
                         "instrument": list(top.index),
                         "score": [float(v) for v in top.values]})


def _load_names() -> dict:
    fp = REPO_ROOT / "production" / "cn_names_cache.json"
    try:
        return json.loads(fp.read_text(encoding="utf-8")).get("map", {}) if fp.exists() else {}
    except Exception:
        return {}


def _served_pred():
    """The newest recorder with a loadable pred.pkl in EXPERIMENT (mirrors backend
    get_latest_recorder_id / daily_inference._find_pooled_recorder)."""
    from production.rolling_train import load_config, init_qlib
    from qlib.workflow import R
    init_qlib(load_config(REPO_ROOT / CONFIG))
    exp = R.get_exp(experiment_name=EXPERIMENT)
    recs = exp.list_recorders()
    recs = list(recs.values()) if isinstance(recs, dict) else recs
    for r in sorted(recs, key=lambda r: r.info.get("start_time", 0), reverse=True):
        try:
            return r.load_object("pred.pkl")
        except Exception:
            continue
    raise RuntimeError(f"no recorder with pred.pkl in {EXPERIMENT}")


def _read_journal() -> pd.DataFrame:
    if JOURNAL.exists():
        return pd.read_csv(JOURNAL, dtype={"instrument": str})
    return pd.DataFrame(columns=COLS)


def snapshot(top_k: int) -> int:
    try:
        pred = _served_pred()
    except Exception as exc:
        print(f"no served prediction available: {exc}")
        return 1
    s = pred["score"] if isinstance(pred, pd.DataFrame) else pred
    as_of = max(s.dropna().index.get_level_values("datetime").unique())
    as_of_str = pd.Timestamp(as_of).date().isoformat()
    j = _read_journal()
    if not j.empty and (j["as_of_date"].astype(str) == as_of_str).any():
        print(f"already journaled as_of={as_of_str} — skip")
        return 0
    names = _load_names()
    top = select_topk(pred, as_of, top_k)
    top["as_of_date"] = as_of_str
    top["name"] = [names.get(str(i)[2:], "") for i in top["instrument"]]
    top["entry_open"] = float("nan")
    top["exit_open"] = float("nan")
    top["realized_ret"] = float("nan")
    top["buyable_at_open"] = ""
    top["reconciled"] = False
    out = pd.concat([j, top[COLS]], ignore_index=True)
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(JOURNAL, index=False)
    print(f"snapshot as_of={as_of_str} top{top_k}: "
          + ", ".join(f"#{int(r['rank'])} {r['instrument']}({r['name']})"
                      for _, r in top.iterrows()))
    print(f"  -> {JOURNAL}")
    return 0


def reconcile() -> int:
    from production.backtest.executability import load_entry_ohlc, buyable_mask
    from production.intraday.exec_backtest import daily_open_adj
    j = _read_journal()
    if j.empty:
        print("empty journal — run `snapshot` first")
        return 0
    pend = j[~j["reconciled"].astype(bool)]
    if pend.empty:
        print("nothing to reconcile")
        return 0
    insts = sorted(pend["instrument"].unique())
    start = str(pd.to_datetime(pend["as_of_date"]).min().date())
    end = str((pd.to_datetime(pend["as_of_date"]).max() + pd.Timedelta(days=40)).date())
    opens = daily_open_adj(insts, start, end, config=CONFIG)
    ohlc = load_entry_ohlc(insts, start, end, config_path=CONFIG)
    buyable = buyable_mask(ohlc)
    dates = sorted(opens.index.get_level_values("datetime").unique())
    pos = {d: i for i, d in enumerate(dates)}
    n_done = 0
    for idx, row in pend.iterrows():
        d = pd.Timestamp(row["as_of_date"])
        inst = str(row["instrument"])
        if d not in pos or pos[d] + 1 + HOLD >= len(dates):
            continue  # decision day missing, or holding period not elapsed yet
        entry_d, exit_d = dates[pos[d] + 1], dates[pos[d] + 1 + HOLD]
        eo, xo = opens.get((entry_d, inst)), opens.get((exit_d, inst))
        if eo is None or xo is None or not (eo > 0):
            continue
        j.loc[idx, "entry_open"] = float(eo)
        j.loc[idx, "exit_open"] = float(xo)
        j.loc[idx, "realized_ret"] = float(xo) / float(eo) - 1
        b = buyable.get((d, inst))
        j.loc[idx, "buyable_at_open"] = "" if b is None else str(bool(b))
        j.loc[idx, "reconciled"] = True
        n_done += 1
    j.to_csv(JOURNAL, index=False)
    rec = j[j["reconciled"].astype(bool)].copy()
    rec["realized_ret"] = pd.to_numeric(rec["realized_ret"], errors="coerce")
    if len(rec):
        per_day = rec.groupby("as_of_date")["realized_ret"].mean().dropna()
        cum = (1 + per_day).prod() - 1 if len(per_day) else float("nan")
        hit = (rec["realized_ret"] > 0).mean()
        print(f"reconciled +{n_done} new; {len(rec)} picks over {per_day.size} days; "
              f"equal-weight cumulative (gross) = {cum:+.2%}; pick hit-rate = {hit:.1%}")
    else:
        print(f"reconciled +{n_done}; none fully elapsed yet")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("snapshot")
    sp.add_argument("--top-k", type=int, default=3)
    sub.add_parser("reconcile")
    args = ap.parse_args()
    return snapshot(args.top_k) if args.cmd == "snapshot" else reconcile()


if __name__ == "__main__":
    raise SystemExit(main())
