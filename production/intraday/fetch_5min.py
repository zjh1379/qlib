"""Fetch raw 5-min bars (+ prev daily close) from baostock, with parquet cache.
Network calls are isolated here; entry_rules/exec_backtest stay pure/offline.

A single baostock session is reused across calls (login once, logout at process
exit) to avoid per-call login churn during large sweeps. prev_close is cached to
JSON so it is fetched once and reused across rules."""
from __future__ import annotations
from pathlib import Path
import atexit
import json
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = REPO_ROOT / "production" / "intraday" / "cache"
_PREV_CLOSE_FP = CACHE / "_prev_close.json"

_logged_in = False
_prev_cache: dict | None = None


def _ensure_login() -> None:
    """Idempotent baostock login; logs out once at process exit."""
    global _logged_in
    if _logged_in:
        return
    import baostock as bs
    bs.login()
    _logged_in = True
    atexit.register(_logout)


def _logout() -> None:
    global _logged_in
    if not _logged_in:
        return
    import baostock as bs
    try:
        bs.logout()
    except Exception:
        pass
    _logged_in = False


def parse_baostock_5min(raw: pd.DataFrame) -> pd.DataFrame:
    """baostock 5min get_data() (all-string cols) -> typed, with parsed `datetime`."""
    df = pd.DataFrame()
    df["datetime"] = pd.to_datetime(raw["time"].str[:14], format="%Y%m%d%H%M%S")
    for c in ("open", "high", "low", "close", "volume", "amount"):
        df[c] = pd.to_numeric(raw[c], errors="coerce")
    return df[["datetime", "open", "high", "low", "close", "volume", "amount"]]


def fetch_5min(instrument: str, start: str, end: str) -> pd.DataFrame:
    """Return typed 5min bars for [start,end] inclusive. Cached per (inst,start,end)
    call as one parquet (per-month sharding is overkill for P1)."""
    from production.intraday.entry_rules import bs_code
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{instrument}_{start}_{end}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    import baostock as bs
    _ensure_login()
    rs = bs.query_history_k_data_plus(
        bs_code(instrument), "time,open,high,low,close,volume,amount",
        start_date=start, end_date=end, frequency="5", adjustflag="3")
    raw = rs.get_data()
    out = parse_baostock_5min(raw) if raw is not None and len(raw) else \
        pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount"])
    out.to_parquet(fp)
    return out


def _prev_close_cache() -> dict:
    global _prev_cache
    if _prev_cache is None:
        if _PREV_CLOSE_FP.exists():
            try:
                _prev_cache = json.loads(_PREV_CLOSE_FP.read_text(encoding="utf-8"))
            except Exception:
                _prev_cache = {}
        else:
            _prev_cache = {}
    return _prev_cache


def prev_close_raw(instrument: str, date: str) -> float | None:
    """Raw (unadjusted) close of the last trading day strictly before `date`,
    via baostock daily — needed for limit-up / gap detection on raw intraday.
    Cached to JSON (key=inst_date); None results are cached too."""
    cache = _prev_close_cache()
    key = f"{instrument}_{date}"
    if key in cache:
        v = cache[key]
        return float(v) if v is not None else None
    from production.intraday.entry_rules import bs_code
    import baostock as bs
    _ensure_login()
    start = (pd.Timestamp(date) - pd.Timedelta(days=12)).date().isoformat()
    rs = bs.query_history_k_data_plus(bs_code(instrument), "date,close",
                                      start_date=start, end_date=date,
                                      frequency="d", adjustflag="3")
    d = rs.get_data()
    val: float | None = None
    if d is not None and len(d) >= 2:
        d = d[d["date"] < date]
        if len(d):
            val = float(d["close"].iloc[-1])
    cache[key] = val
    CACHE.mkdir(parents=True, exist_ok=True)
    _PREV_CLOSE_FP.write_text(json.dumps(cache), encoding="utf-8")
    return val
