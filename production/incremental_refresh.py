"""Incremental multi-market data refresh: only fetch missing days per symbol.

Emits structured progress to stdout for the backend to parse:
    PROGRESS {"phase":"fetch","current":42,"total":300,"message":"sh.600519: +3 rows"}

Phases (in order):
    init       -> baostock login + getting market constituent lists
    fetch      -> per-symbol incremental OHLCV fetch (across all selected markets)
    dump       -> dump_bin dump_update (single op, current=1 total=1 at start, both=1 at end)
    benchmark  -> SH000300 fetch + dump
    done       -> finished

Usage:
    python production/incremental_refresh.py
        [--csv_dir <dir>] (default: ~/.qlib/stock_data/csi300_csv)
        [--qlib_dir <dir>] (default: ~/.qlib/qlib_data/cn_data_bs)
        [--start <YYYY-MM-DD>] (default: 2018-01-01, used only when CSV missing)
        [--full] force full re-fetch (skip incremental check)
        [--markets csi300,csi500,etfs,custom] comma-separated subset (default: all four)
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


# Hot ETFs (baostock codes). Verify each one fetches successfully; log warnings + skip failures.
# Organized by theme for clarity. Add new themes / codes here.
ETF_LIST: list[tuple[str, str]] = [
    # === 宽基指数 ===
    ("sh.510300", "沪深300ETF"),
    ("sh.510500", "中证500ETF"),
    ("sh.510050", "上证50ETF"),
    ("sh.510880", "红利ETF"),
    ("sh.510180", "上证180ETF"),
    ("sh.588000", "科创50ETF"),
    ("sh.560090", "中证1000ETF"),
    ("sh.560060", "中证A50ETF"),
    ("sz.159915", "创业板ETF"),
    ("sz.159949", "创业板50ETF"),

    # === 半导体 / 芯片 ===
    ("sz.159995", "芯片ETF华夏"),
    ("sz.159599", "芯片ETF东财"),
    ("sh.512480", "半导体ETF"),
    ("sh.512760", "半导体50ETF"),
    ("sh.516920", "集成电路ETF"),
    ("sz.159801", "半导体设备ETF"),

    # === 人工智能 / 科技 ===
    ("sh.515980", "人工智能AIETF"),
    ("sh.512720", "计算机ETF"),
    ("sh.515050", "5GETF"),

    # === 机器人 / 智能制造 ===
    ("sz.159770", "机器人ETF"),

    # === 航天 / 军工 ===
    ("sh.512710", "军工龙头ETF"),
    ("sh.512560", "军工ETF"),
    ("sz.159518", "航天航空ETF"),

    # === 医药 / 生物 ===
    ("sh.512170", "医疗ETF"),
    ("sh.512290", "生物医药ETF"),
    ("sh.515210", "创新药ETF"),

    # === 新能源 / 锂电 / 光伏 ===
    ("sh.515030", "新能源车ETF"),
    ("sh.515790", "光伏ETF"),
    ("sz.159875", "锂电池ETF"),
    ("sz.159611", "电力ETF"),

    # === 消费 / 食品 / 酒 ===
    ("sz.159928", "消费ETF"),
    ("sh.512690", "酒ETF"),
    ("sh.515170", "食品饮料ETF"),

    # === 金融 / 房地产 ===
    ("sh.512880", "证券ETF"),
    ("sh.512800", "银行ETF"),
    ("sh.512000", "券商ETF"),
    ("sh.512200", "房地产ETF"),

    # === 港股 / 海外 ===
    ("sh.513050", "中概互联ETF"),
    ("sh.513180", "恒生科技ETF"),
    ("sh.513060", "恒生医药ETF"),

    # === 商品 / 黄金 ===
    ("sh.518880", "黄金ETF"),
]

MARKETS: dict[str, str] = {
    "csi300": "沪深300",
    "csi500": "中证500",
    "etfs": "热门ETF",
    "custom": "自定义",
}


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


def get_csi500_codes() -> list[str]:
    rs = bs.query_zz500_stocks()
    if rs.error_code != "0":
        raise RuntimeError(f"query_zz500_stocks failed: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    return df["code"].tolist()


def get_etf_codes() -> list[str]:
    return [code for code, _name in ETF_LIST]


def write_etf_names_json(repo_root: Path) -> None:
    """Sync production/etf_names.json from ETF_LIST so the frontend sees
    Chinese names. Single source of truth = ETF_LIST in this file."""
    out: dict[str, str] = {}
    for code, name in ETF_LIST:
        # baostock 'sh.510300' -> qlib 'SH510300'
        market, num = code.split(".")
        qlib_sym = market.upper() + num
        out[qlib_sym] = name
    target = repo_root / "production" / "etf_names.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_custom_codes(repo_root: Path) -> list[str]:
    """Read production/custom_symbols.txt; convert SH600519-format to sh.600519 for baostock."""
    f = repo_root / "production" / "custom_symbols.txt"
    if not f.is_file():
        return []
    codes = []
    for line in f.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Accept either SH600519 or sh.600519 format
        if s.startswith(("SH", "SZ")) and len(s) == 8:
            codes.append(s[:2].lower() + "." + s[2:])
        elif "." in s:
            codes.append(s.lower())
    return codes


def write_market_instruments(qlib_dir: Path, csv_dir: Path, market_name: str, qlib_symbols: list[str]) -> None:
    """Write instruments/{market_name}.txt by inspecting each symbol's CSV for first/last date.

    Format: SYMBOL\tFIRST_DATE\tLAST_DATE  (qlib convention)
    """
    out_path = qlib_dir / "instruments" / f"{market_name}.txt"
    lines = []
    for sym in qlib_symbols:
        csv = csv_dir / f"{sym}.csv"
        if not csv.is_file():
            continue
        first = None
        last = None
        try:
            with csv.open("r", encoding="utf-8") as f:
                _header = f.readline()  # skip
                for line in f:
                    parts = line.split(",")
                    if len(parts) < 2:
                        continue
                    if first is None:
                        first = parts[1]
                    last = parts[1]
        except Exception:
            continue
        if first and last:
            lines.append(f"{sym}\t{first}\t{last}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", default=os.path.expanduser("~/.qlib/stock_data/csi300_csv"))
    p.add_argument("--qlib_dir", default=os.path.expanduser("~/.qlib/qlib_data/cn_data_bs"))
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--full", action="store_true", help="ignore incremental check")
    p.add_argument(
        "--markets",
        default="csi300,csi500,etfs,custom",
        help="comma-separated subset of csi300,csi500,etfs,custom",
    )
    args = p.parse_args()

    selected = [m.strip() for m in args.markets.split(",") if m.strip() in MARKETS]
    csv_dir = Path(args.csv_dir)
    qlib_dir = Path(args.qlib_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent

    # Keep production/etf_names.json in sync with ETF_LIST so the frontend
    # always sees up-to-date Chinese names, even before any data is fetched.
    write_etf_names_json(repo_root)

    progress("init", 0, 1, "logging in to baostock")
    bs.login()
    try:
        # 1. Build per-market code lists
        progress("init", 1, 1, "fetching market constituent lists")
        market_codes: dict[str, list[str]] = {}
        if "csi300" in selected:
            market_codes["csi300"] = get_csi300_codes()
        if "csi500" in selected:
            market_codes["csi500"] = get_csi500_codes()
        if "etfs" in selected:
            market_codes["etfs"] = get_etf_codes()
        if "custom" in selected:
            market_codes["custom"] = get_custom_codes(repo_root)

        # 2. Union with provenance: a code may appear in multiple markets; we fetch once.
        all_codes: list[str] = []
        seen: set[str] = set()
        market_to_qlib_syms: dict[str, list[str]] = {m: [] for m in market_codes}
        for m, codes in market_codes.items():
            for c in codes:
                qs = to_qlib_symbol(c)
                market_to_qlib_syms[m].append(qs)
                if c not in seen:
                    seen.add(c)
                    all_codes.append(c)

        today_str = date.today().isoformat()
        progress("fetch", 0, len(all_codes), f"start incremental fetch to {today_str}")

        total_appended = 0
        for i, bs_code in enumerate(all_codes, 1):
            sym = to_qlib_symbol(bs_code)
            target = csv_dir / f"{sym}.csv"
            start = args.start
            if not args.full:
                last = read_csv_last_date(target)
                if last is not None:
                    start = (last + timedelta(days=1)).isoformat()
            if date.fromisoformat(start) > date.fromisoformat(today_str):
                progress("fetch", i, len(all_codes), f"{bs_code}: already up to date")
                continue
            try:
                df = fetch_one(bs_code, start, today_str)
            except Exception as e:
                progress("fetch", i, len(all_codes), f"{bs_code}: ERROR {e}")
                continue
            n = append_to_csv(target, sym, df) if not df.empty else 0
            total_appended += n
            progress("fetch", i, len(all_codes), f"{bs_code}: +{n} rows")

        # 3. dump_bin update
        progress("dump", 0, 1, f"updating qlib bins (~{total_appended} new rows)")
        run_dump_update(repo_root, csv_dir, qlib_dir)
        progress("dump", 1, 1, "bins updated")

        # 4. Per-market instruments files
        for m, qsyms in market_to_qlib_syms.items():
            if qsyms:
                write_market_instruments(qlib_dir, csv_dir, m, qsyms)

        # 5. Benchmark
        progress("benchmark", 0, 1, "fetching SH000300 benchmark")
        bench_rows = refresh_benchmark(csv_dir, repo_root, qlib_dir)
        progress("benchmark", 1, 1, f"benchmark +{bench_rows} rows")

        # 6. Final calendar end
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
        progress("done", 1, 1, f"完成 · calendar_end={cal_end} · markets={','.join(selected)}")
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
