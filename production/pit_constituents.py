"""Point-in-time CSI300 + CSI500 constituents.

Pulls monthly snapshots from baostock; caches to parquet.
Fail-soft: if remote fetch fails, falls back to the cached file as long as
the cache is no older than `allow_stale_days`.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

_log = logging.getLogger(__name__)

DEFAULT_CACHE = Path(__file__).resolve().parent / "pit_constituents.parquet"

# Acceptable membership-count ranges per monthly snapshot
CSI300_RANGE = (280, 320)
CSI500_RANGE = (480, 520)


def _is_within_range(n: int, rng: tuple[int, int]) -> bool:
    lo, hi = rng
    return lo <= n <= hi


def _month_starts(start: date, end: date) -> list[date]:
    out: list[date] = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        out.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _fetch_remote(snapshot_dates: list[date]) -> pd.DataFrame:
    """Hit baostock for each (date, index). Returns the unioned long df."""
    import baostock as bs

    bs.login()
    rows: list[dict] = []
    try:
        for d in snapshot_dates:
            ds = d.strftime("%Y-%m-%d")
            for query_fn, label in (
                (bs.query_hs300_stocks, "csi300"),
                (bs.query_zz500_stocks, "csi500"),
            ):
                rs = query_fn(date=ds)
                while rs.error_code == "0" and rs.next():
                    row = dict(zip(rs.fields, rs.get_row_data()))
                    code = row["code"]
                    qcode = _bs_to_qlib(code)
                    rows.append({"snapshot_date": d, "instrument": qcode, "membership": label})
    finally:
        bs.logout()
    return pd.DataFrame(rows)


def _bs_to_qlib(bs_code: str) -> str:
    """Convert 'sh.600000' -> 'SH600000', 'sz.000001' -> 'SZ000001'."""
    parts = bs_code.split(".")
    return f"{parts[0].upper()}{parts[1]}"


def load_or_refresh(
    end: date,
    cache_path: Path = DEFAULT_CACHE,
    allow_stale_days: int = 30,
    history_years: int = 7,
) -> pd.DataFrame:
    """Return the PIT df (long format: snapshot_date, instrument, membership).

    Refresh strategy:
      - If cache exists and its newest snapshot is within `allow_stale_days`, return cache.
      - Otherwise refetch the entire history_years window and overwrite cache.
      - If refetch fails, return cache anyway and log a warning.
    """
    cached: pd.DataFrame | None = None
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            if cached.empty:
                _log.warning("pit_cache_empty_refetching")
                cached = None
            else:
                cached_max = pd.to_datetime(cached["snapshot_date"]).dt.date.max()
                if (end - cached_max).days <= allow_stale_days:
                    _log.info("pit_cache_hit", extra={"max_date": str(cached_max)})
                    return cached
        except Exception as exc:
            _log.warning("pit_cache_corrupt_refetching", extra={"error": str(exc)})
            cached = None

    start = date(end.year - history_years, end.month, 1)
    months = _month_starts(start, end)
    try:
        fresh = _fetch_remote(months)
    except Exception as exc:
        if cached is not None:
            _log.warning("pit_fetch_failed_using_cache", extra={"error": str(exc)})
            return cached
        raise

    # Sanity check the most recent snapshot
    last = fresh[fresh["snapshot_date"] == fresh["snapshot_date"].max()]
    n300 = (last["membership"] == "csi300").sum()
    n500 = (last["membership"] == "csi500").sum()
    if not _is_within_range(n300, CSI300_RANGE) or not _is_within_range(n500, CSI500_RANGE):
        msg = f"pit_constituents_undersized: csi300={n300}, csi500={n500}"
        _log.warning(msg)
        if cached is not None:
            return cached
        raise RuntimeError(msg)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_parquet(cache_path)
    _log.info("pit_refresh_ok", extra={"rows": len(fresh)})
    return fresh


def members_on(df: pd.DataFrame, query_date: date) -> list[str]:
    """Return the instruments that were CSI300 or CSI500 members on `query_date`
    (using the most recent month-start snapshot <= query_date)."""
    # Normalize the snapshot_date column to Timestamp once, then compare on the
    # date portion. This works whether the underlying values are Python date,
    # pandas Timestamp, or datetime64.
    dates_ts = pd.to_datetime(df["snapshot_date"])
    q_ts = pd.Timestamp(query_date)
    eligible = dates_ts[dates_ts <= q_ts]
    if eligible.empty:
        return []
    cutoff = eligible.max()
    return df.loc[dates_ts == cutoff, "instrument"].unique().tolist()


def _read_market_file(qlib_data_root: Path, market: str) -> list[str]:
    """Read an existing qlib instruments file (e.g. 'etfs', 'custom').
    Returns the list of symbol codes. Skips silently if file missing or
    unreadable."""
    p = qlib_data_root / "instruments" / f"{market}.txt"
    if not p.exists():
        return []
    try:
        out: list[str] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            tok = line.split("\t", 1)
            if tok and tok[0].strip():
                out.append(tok[0].strip())
        return out
    except Exception as exc:
        _log.warning("read_market_file_failed market=%s error=%s", market, exc)
        return []


def write_pit_instruments_file(
    df: pd.DataFrame,
    end_date: date,
    name: str,
    qlib_data_root: Path,
    lookback_years: int = 7,
    extra_markets: list[str] | None = None,
) -> Path:
    """Write a qlib-compatible instruments file at
    `<qlib_data_root>/instruments/<name>.txt`.

    Format (TSV, no header):
        <instrument>\t<start_date>\t<end_date>

    The set of instruments is the **union** of all stocks that were members of
    CSI300 or CSI500 at any monthly snapshot within
    `[end_date - lookback_years, end_date]`. Each row spans that whole window —
    qlib does not natively support overlapping per-stock PIT ranges, so we
    accept this approximation. True per-day PIT filtering happens upstream in
    rolling_train.run_once by reindexing training samples against the long
    `df` returned by `load_or_refresh()`.

    Returns the absolute path of the written file.
    """
    start_date = date(end_date.year - lookback_years, end_date.month, 1)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    dates_ts = pd.to_datetime(df["snapshot_date"])
    window_mask = (dates_ts >= start_ts) & (dates_ts <= end_ts)
    members = sorted(df.loc[window_mask, "instrument"].unique().tolist())

    # Optionally union with non-PIT extras (ETFs, custom symbols). These
    # don't have monthly PIT snapshots — we use the same window for all
    # symbols in the extras list. The training pipeline still per-day
    # filters via members_on(...) for csi800 members; extras pass through
    # because they're always "members" (not subject to PIT filtering).
    extras: set[str] = set()
    if extra_markets:
        for market in extra_markets:
            for sym in _read_market_file(qlib_data_root, market):
                if sym not in members:
                    extras.add(sym)
        extras -= set(members)  # dedup

    all_members = sorted(set(members) | extras)

    out_dir = qlib_data_root / "instruments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.txt"
    start_iso = start_date.isoformat()
    end_iso = end_date.isoformat()
    lines = [f"{inst}\t{start_iso}\t{end_iso}" for inst in all_members]
    out_path.write_text("\n".join(lines) + "\n")
    _log.info(
        "pit_instruments_written path=%s count=%d name=%s extras=%d",
        str(out_path), len(all_members), name, len(extras),
    )
    return out_path
