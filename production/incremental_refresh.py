"""Incremental CSI300 data refresh: only fetch missing days per stock.

Emits structured progress to stdout for the backend to parse:
    PROGRESS {"phase":"fetch","current":42,"total":300,"message":"sh.600519: +3 rows"}

Phases (in order):
    init       -> baostock login + getting CSI300 list
    fetch      -> per-stock incremental OHLCV fetch
    dump       -> dump_bin dump_update (single op, current=1 total=1 at start, both=1 at end)
    benchmark  -> SH000300 fetch + dump
    done       -> finished

Usage:
    python production/incremental_refresh.py
        [--csv_dir <dir>] (default: ~/.qlib/stock_data/csi300_csv)
        [--qlib_dir <dir>] (default: ~/.qlib/qlib_data/cn_data_bs)
        [--start <YYYY-MM-DD>] (default: 2018-01-01, used only when CSV missing)
        [--full] force full re-fetch (skip incremental check)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

# CN sources are domestic; clear dead proxy if any
for _v in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY']:
    os.environ.pop(_v, None)

import pandas as pd
import baostock as bs


def progress(phase: str, current: int, total: int, message: str = "") -> None:
    """Emit a structured progress line that the backend can parse."""
    line = json.dumps(
        {"phase": phase, "current": current, "total": total, "message": message},
        ensure_ascii=False,
    )
    print(f"PROGRESS {line}", flush=True)


def to_qlib_symbol(bs_code: str) -> str:
    """sh.600519 -> SH600519; sz.000001 -> SZ000001."""
    market, num = bs_code.split(".")
    return market.upper() + num


def read_csv_last_date(csv_path: Path) -> date | None:
    """Read the last (most recent) date in a CSV. Returns None if file missing or unreadable."""
    if not csv_path.is_file():
        return None
    try:
        # Read just the last bytes for efficiency on large CSVs
        with csv_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = min(size, 4096)
            f.seek(size - block)
            tail = f.read().decode("utf-8", errors="replace")
        lines = [ln for ln in tail.splitlines() if ln.strip()]
        if not lines:
            return None
        last = lines[-1]
        # CSV format: symbol,date,open,high,low,close,volume,factor,change
        parts = last.split(",")
        if len(parts) < 2:
            return None
        return date.fromisoformat(parts[1])
    except Exception as e:
        print(f"  [warn] could not read last date from {csv_path.name}: {e}", file=sys.stderr)
        return None


def fetch_one(bs_code: str, start: str, end: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,open,high,low,close,volume,pctChg",
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag="2",  # 2 = qfq forward-adjusted
    )
    if rs.error_code != "0":
        raise RuntimeError(f"{bs_code}: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=rs.fields)
    for c in ["open", "high", "low", "close", "volume", "pctChg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "close"])
    df["change"] = df["pctChg"] / 100.0
    return df.drop(columns=["pctChg"])


def append_to_csv(csv_path: Path, sym: str, df: pd.DataFrame) -> int:
    df = df.copy()
    df.insert(0, "symbol", sym)
    df["factor"] = 1.0
    df = df[["symbol", "date", "open", "high", "low", "close", "volume", "factor", "change"]]
    header_needed = not csv_path.is_file() or csv_path.stat().st_size == 0
    df.to_csv(csv_path, mode="a", header=header_needed, index=False)
    return len(df)


def get_csi300_codes() -> list[str]:
    rs = bs.query_hs300_stocks()
    if rs.error_code != "0":
        raise RuntimeError(f"query_hs300_stocks failed: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    return df["code"].tolist()


def run_dump_update(repo_root: Path, csv_dir: Path, qlib_dir: Path) -> None:
    """Call scripts/dump_bin.py dump_update via subprocess."""
    dump_bin = repo_root / "scripts" / "dump_bin.py"
    cmd = [
        sys.executable,
        str(dump_bin),
        "dump_update",
        "--data_path", str(csv_dir),
        "--qlib_dir", str(qlib_dir),
        "--freq", "day",
        "--exclude_fields", "symbol,date",
        "--include_fields", "open,close,high,low,volume,factor,change",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"dump_bin dump_update failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
        )


def refresh_benchmark(csv_dir: Path, repo_root: Path, qlib_dir: Path) -> int:
    """Fetch sh.000300, write CSV, then dump it.

    Returns rows fetched (for logging).
    """
    target = csv_dir / "SH000300.csv"
    start = "2018-01-01"
    last_date = read_csv_last_date(target)
    if last_date is not None:
        start = (last_date + timedelta(days=1)).isoformat()
    today_str = date.today().isoformat()
    if date.fromisoformat(start) > date.fromisoformat(today_str):
        return 0
    rs = bs.query_history_k_data_plus(
        "sh.000300",
        "date,open,high,low,close,volume,pctChg",
        start_date=start, end_date=today_str, frequency="d", adjustflag="3",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"sh.000300 fetch failed: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=rs.fields)
    for c in ["open", "high", "low", "close", "volume", "pctChg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "close"])
    df["change"] = df["pctChg"] / 100.0
    df = df.drop(columns=["pctChg"])
    n = append_to_csv(target, "SH000300", df)
    # dump_fix adds new symbols if missing; dump_update appends new dates to existing.
    # If SH000300 was already in qlib_dir we want dump_update; if it's first time, dump_fix.
    if (qlib_dir / "features" / "sh000300").is_dir():
        # already there: dump_update is the right choice (rebuilds all symbols in csv_dir)
        run_dump_update(repo_root, csv_dir, qlib_dir)
    else:
        # first time: dump_fix
        dump_bin = repo_root / "scripts" / "dump_bin.py"
        cmd = [
            sys.executable, str(dump_bin), "dump_fix",
            "--data_path", str(csv_dir),
            "--qlib_dir", str(qlib_dir),
            "--freq", "day",
            "--exclude_fields", "symbol,date",
            "--include_fields", "open,close,high,low,volume,factor,change",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"dump_fix failed (rc={result.returncode}): {result.stderr[-1000:]}")
    return n


def write_csi300_instruments(qlib_dir: Path) -> None:
    """Build instruments/csi300.txt from instruments/all.txt (excluding SH000300 if present)."""
    all_file = qlib_dir / "instruments" / "all.txt"
    csi_file = qlib_dir / "instruments" / "csi300.txt"
    if not all_file.is_file():
        return
    lines = [ln for ln in all_file.read_text(encoding="utf-8").splitlines() if not ln.startswith("SH000300")]
    csi_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", default=os.path.expanduser("~/.qlib/stock_data/csi300_csv"))
    p.add_argument("--qlib_dir", default=os.path.expanduser("~/.qlib/qlib_data/cn_data_bs"))
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--full", action="store_true", help="ignore incremental check")
    args = p.parse_args()

    csv_dir = Path(args.csv_dir)
    qlib_dir = Path(args.qlib_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent

    progress("init", 0, 1, "logging in to baostock")
    bs.login()
    try:
        progress("init", 1, 1, "fetching CSI300 list")
        codes = get_csi300_codes()
        today_str = date.today().isoformat()

        progress("fetch", 0, len(codes), f"start incremental fetch to {today_str}")
        total_appended = 0
        skipped_uptodate = 0
        for i, bs_code in enumerate(codes, 1):
            sym = to_qlib_symbol(bs_code)
            target = csv_dir / f"{sym}.csv"
            start = args.start
            if not args.full:
                last = read_csv_last_date(target)
                if last is not None:
                    start = (last + timedelta(days=1)).isoformat()
            if date.fromisoformat(start) > date.fromisoformat(today_str):
                skipped_uptodate += 1
                progress("fetch", i, len(codes), f"{bs_code}: already up to date")
                continue
            try:
                df = fetch_one(bs_code, start, today_str)
            except Exception as e:
                progress("fetch", i, len(codes), f"{bs_code}: ERROR {e}")
                continue
            n = append_to_csv(target, sym, df) if not df.empty else 0
            total_appended += n
            progress("fetch", i, len(codes), f"{bs_code}: +{n} rows")

        progress(
            "dump", 0, 1,
            f"updating qlib bins (appended ~{total_appended} rows total, {skipped_uptodate} stocks already current)",
        )
        if total_appended > 0 or skipped_uptodate < len(codes):
            run_dump_update(repo_root, csv_dir, qlib_dir)
        progress("dump", 1, 1, "bins updated")

        progress("benchmark", 0, 1, "fetching SH000300 benchmark")
        bench_rows = refresh_benchmark(csv_dir, repo_root, qlib_dir)
        progress("benchmark", 1, 1, f"benchmark +{bench_rows} rows")

        write_csi300_instruments(qlib_dir)

        # Read final calendar end for the success message
        cal_file = qlib_dir / "calendars" / "day.txt"
        cal_end = "?"
        if cal_file.is_file():
            try:
                with cal_file.open("rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - 256))
                    last = f.read().decode("utf-8", errors="replace").strip().splitlines()
                    if last:
                        cal_end = last[-1]
            except Exception:
                pass
        progress("done", 1, 1, f"完成 · calendar_end={cal_end}")
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
