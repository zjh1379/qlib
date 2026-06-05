"""Fetch raw 5-min bars (+ prev daily close) from baostock, with parquet cache.
Network calls are isolated here; entry_rules/exec_backtest stay pure/offline."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = REPO_ROOT / "production" / "intraday" / "cache"


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
    bs.login()
    try:
        rs = bs.query_history_k_data_plus(
            bs_code(instrument), "time,open,high,low,close,volume,amount",
            start_date=start, end_date=end, frequency="5", adjustflag="3")
        raw = rs.get_data()
    finally:
        bs.logout()
    out = parse_baostock_5min(raw) if raw is not None and len(raw) else \
        pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount"])
    out.to_parquet(fp)
    return out


def prev_close_raw(instrument: str, date: str) -> float | None:
    """Raw (unadjusted) close of the last trading day strictly before `date`,
    via baostock daily — needed for limit-up / gap detection on raw intraday."""
    from production.intraday.entry_rules import bs_code
    start = (pd.Timestamp(date) - pd.Timedelta(days=12)).date().isoformat()
    import baostock as bs
    bs.login()
    try:
        rs = bs.query_history_k_data_plus(bs_code(instrument), "date,close",
                                          start_date=start, end_date=date,
                                          frequency="d", adjustflag="3")
        d = rs.get_data()
    finally:
        bs.logout()
    if d is None or len(d) < 2:
        return None
    d = d[d["date"] < date]
    return float(d["close"].iloc[-1]) if len(d) else None
