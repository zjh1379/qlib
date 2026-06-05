"""Fetch raw 5-min bars (+ prev daily close) from baostock, with parquet cache.
Network calls are isolated here; entry_rules/exec_backtest stay pure/offline.

Robustness: baostock drops its socket after many rapid queries. We keep one warm
session but verify every login/query `error_code`; on any failure we force a clean
re-login and retry. Results are cached ONLY on a genuine query success (error_code
'0') — a 0-row success caches as real "no-data" (halt), while a connection failure
returns empty WITHOUT caching, so a re-run transparently resumes."""
from __future__ import annotations
from pathlib import Path
import atexit
import json
import time
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = REPO_ROOT / "production" / "intraday" / "cache"
_PREV_CLOSE_FP = CACHE / "_prev_close.json"
_COLS = ["datetime", "open", "high", "low", "close", "volume", "amount"]

_logged_in = False
_prev_cache: dict | None = None


def _do_login() -> bool:
    import baostock as bs
    lg = bs.login()
    return lg is not None and getattr(lg, "error_code", "1") == "0"


def _ensure_login() -> None:
    """Idempotent baostock login (verified); logs out once at process exit."""
    global _logged_in
    if _logged_in:
        return
    for attempt in range(4):
        if _do_login():
            _logged_in = True
            atexit.register(_logout)
            return
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("baostock login failed after retries")


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


def _force_relogin() -> None:
    global _logged_in
    import baostock as bs
    try:
        bs.logout()
    except Exception:
        pass
    _logged_in = False


def _query(make_rs, max_try: int = 4):
    """make_rs: fn(bs) -> ResultData. Returns get_data() DataFrame on success
    (error_code '0'), or None if the connection failed after retries."""
    import baostock as bs
    for attempt in range(max_try):
        _ensure_login()
        try:
            rs = make_rs(bs)
            if rs is not None and getattr(rs, "error_code", "1") == "0":
                return rs.get_data()
        except Exception:
            pass
        _force_relogin()                       # drop the dead socket; next loop re-logins
        time.sleep(0.5 * (attempt + 1))
    return None


def parse_baostock_5min(raw: pd.DataFrame) -> pd.DataFrame:
    """baostock 5min get_data() (all-string cols) -> typed, with parsed `datetime`."""
    df = pd.DataFrame()
    df["datetime"] = pd.to_datetime(raw["time"].str[:14], format="%Y%m%d%H%M%S")
    for c in ("open", "high", "low", "close", "volume", "amount"):
        df[c] = pd.to_numeric(raw[c], errors="coerce")
    return df[_COLS]


def fetch_5min(instrument: str, start: str, end: str) -> pd.DataFrame:
    """Return typed 5min bars for [start,end] inclusive, cached per (inst,start,end)
    parquet. Connection failure -> empty DataFrame, NOT cached (re-tryable)."""
    from production.intraday.entry_rules import bs_code
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{instrument}_{start}_{end}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    df = _query(lambda bs: bs.query_history_k_data_plus(
        bs_code(instrument), "time,open,high,low,close,volume,amount",
        start_date=start, end_date=end, frequency="5", adjustflag="3"))
    if df is None:
        return pd.DataFrame(columns=_COLS)     # connection failed -> do not cache
    out = parse_baostock_5min(df) if len(df) else pd.DataFrame(columns=_COLS)
    out.to_parquet(fp)                         # success (incl. genuine 0-row no-data)
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
    via baostock daily. Cached to JSON (key=inst_date). Connection failure ->
    None and NOT cached (re-tryable); a genuine 'no prior day' caches as null."""
    cache = _prev_close_cache()
    key = f"{instrument}_{date}"
    if key in cache:
        v = cache[key]
        return float(v) if v is not None else None
    from production.intraday.entry_rules import bs_code
    start = (pd.Timestamp(date) - pd.Timedelta(days=12)).date().isoformat()
    d = _query(lambda bs: bs.query_history_k_data_plus(
        bs_code(instrument), "date,close", start_date=start, end_date=date,
        frequency="d", adjustflag="3"))
    if d is None:
        return None                            # connection failed -> do not cache
    val: float | None = None
    if len(d) >= 1:
        d = d[d["date"] < date]
        if len(d):
            val = float(d["close"].iloc[-1])
    cache[key] = val
    CACHE.mkdir(parents=True, exist_ok=True)
    _PREV_CLOSE_FP.write_text(json.dumps(cache), encoding="utf-8")
    return val
