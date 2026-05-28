from __future__ import annotations

import json as _json
import re as _re
import subprocess
import sys as _sys
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.core.exceptions import ConflictError, DependencyError, NotFoundError
from app.core.qlib_adapter import (
    get_calendar_info,
    get_csi300_instruments,
    get_csi300_with_names,
    get_instruments_for_market,
    get_market_with_names,
    list_available_markets,
)
from app.data.schemas import (
    AddSymbolResponse,
    DataStatus,
    InstrumentItem,
    InstrumentsResponse,
    MarketInfo,
    MarketsResponse,
    ProgressInfo,
    RefreshJobStatus,
    RefreshResponse,
)

_VALID_SYMBOL_RE = _re.compile(r"^(SH|SZ)\d{6}$")

# Module-level state for refresh jobs
_refresh_jobs: dict[str, dict[str, Any]] = {}
_running_lock = threading.Lock()
# Track which job currently holds the running slot
_running_job_id: str | None = None
_running_state_lock = threading.Lock()


def _calendar_day_file() -> Path:
    """Path to ~/.qlib/qlib_data/cn_data_bs/calendars/day.txt."""
    return Path("~/.qlib/qlib_data/cn_data_bs/calendars/day.txt").expanduser()


def _worktree_root() -> Path:
    """Resolve the worktree root: service.py -> data/ -> app/ -> backend/ -> root."""
    return Path(__file__).resolve().parents[3]


def _last_business_day(today: date) -> date:
    """Return the last expected business day on or before `today` (no holiday calendar)."""
    # weekday(): Mon=0 .. Sun=6
    if today.weekday() == 5:  # Sat -> Fri
        return today - timedelta(days=1)
    if today.weekday() == 6:  # Sun -> Fri
        return today - timedelta(days=2)
    return today


def _business_days_between(start: date, end: date) -> int:
    """Approximate business-day count between two dates (end - start), excluding weekends.

    Returns 0 if end <= start.
    """
    if end <= start:
        return 0
    days = 0
    d = start
    while d < end:
        d = d + timedelta(days=1)
        if d.weekday() < 5:
            days += 1
    return days


def _compute_freshness(calendar_end: date, today: date) -> str:
    expected = _last_business_day(today)
    if calendar_end >= expected:
        return "fresh"
    behind = _business_days_between(calendar_end, expected)
    if behind <= 1:
        return "stale_1d"
    return "stale_2d_plus"


def get_data_status() -> DataStatus:
    cal_end, cal_size = get_calendar_info()
    instruments = get_csi300_instruments()
    day_file = _calendar_day_file()
    last_refresh_at: str | None = None
    if day_file.is_file():
        mtime = datetime.fromtimestamp(day_file.stat().st_mtime, tz=timezone.utc)
        last_refresh_at = mtime.isoformat()
    today = datetime.now().date()
    freshness = _compute_freshness(cal_end, today)
    return DataStatus(
        calendar_end=str(cal_end),
        calendar_size=cal_size,
        instruments_count=len(instruments),
        last_refresh_at=last_refresh_at,
        freshness=freshness,
    )


def get_csi300() -> InstrumentsResponse:
    raw = get_csi300_with_names()
    items = [InstrumentItem(symbol=d["symbol"], name=d["name"]) for d in raw]
    return InstrumentsResponse(market="csi300", count=len(items), items=items)


def list_markets() -> MarketsResponse:
    items = list_available_markets()
    markets = [MarketInfo(**it) for it in items]
    # total: union count (dedup across markets)
    union: set[str] = set()
    for m in markets:
        union.update(get_instruments_for_market(m.name))
    return MarketsResponse(markets=markets, total=len(union))


def list_instruments_for(market: str) -> InstrumentsResponse:
    """Replaces the old get_csi300() for any market name. 'all' returns the union."""
    if market == "all":
        union_set: dict[str, dict] = {}
        for m_info in list_available_markets():
            for it in get_market_with_names(m_info["name"]):
                if it["symbol"] not in union_set:
                    union_set[it["symbol"]] = it
        items = [
            InstrumentItem(symbol=d["symbol"], name=d["name"])
            for d in sorted(union_set.values(), key=lambda x: x["symbol"])
        ]
        return InstrumentsResponse(market=market, count=len(items), items=items)

    raw = get_market_with_names(market)
    if not raw:
        raise NotFoundError(
            f"market '{market}' not found or empty",
            code="market_missing",
            context={"market": market},
        )
    items = [InstrumentItem(symbol=d["symbol"], name=d["name"]) for d in raw]
    return InstrumentsResponse(market=market, count=len(items), items=items)


def add_custom_symbol(symbol: str) -> AddSymbolResponse:
    """Validate, persist to production/custom_symbols.txt, fetch sync via baostock, dump_bin update."""
    sym = symbol.strip().upper()
    if not _VALID_SYMBOL_RE.match(sym):
        raise ConflictError(
            f"symbol must be SH###### or SZ######, got '{symbol}'",
            code="invalid_symbol",
            context={"symbol": symbol},
        )

    # Persist to custom_symbols.txt
    repo_root = _worktree_root()
    custom_file = repo_root / "production" / "custom_symbols.txt"
    custom_file.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if custom_file.is_file():
        for line in custom_file.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                existing.add(s.upper())
    if sym not in existing:
        with custom_file.open("a", encoding="utf-8") as f:
            f.write(sym + "\n")

    # Synchronous fetch via baostock + dump_bin (single symbol)
    bs_code = sym[:2].lower() + "." + sym[2:]
    import baostock as bs
    bs.login()
    rows_appended = 0
    try:
        csv_dir = Path("~/.qlib/stock_data/csi300_csv").expanduser()
        csv_dir.mkdir(parents=True, exist_ok=True)
        target = csv_dir / f"{sym}.csv"

        # Reuse fetch helpers from the production script
        import sys as _sys_local
        prod_dir = repo_root / "production"
        if str(prod_dir) not in _sys_local.path:
            _sys_local.path.insert(0, str(prod_dir))
        from incremental_refresh import (  # type: ignore
            append_to_csv as _append_to_csv,
            fetch_one as _fetch_one,
            read_csv_last_date as _read_csv_last_date,
        )

        start = "2018-01-01"
        last = _read_csv_last_date(target)
        if last is not None:
            start = (last + timedelta(days=1)).isoformat()
        today_str = date.today().isoformat()
        if date.fromisoformat(start) <= date.fromisoformat(today_str):
            df = _fetch_one(bs_code, start, today_str)
            if not df.empty:
                rows_appended = _append_to_csv(target, sym, df)
    finally:
        bs.logout()

    # dump_bin update on csv_dir (whole dir, but dump_update is incremental per file)
    qlib_dir = Path("~/.qlib/qlib_data/cn_data_bs").expanduser()
    dump_bin = repo_root / "scripts" / "dump_bin.py"
    csv_dir = Path("~/.qlib/stock_data/csi300_csv").expanduser()
    proc = subprocess.run(
        [
            _sys.executable,
            str(dump_bin),
            "dump_update",
            "--data_path", str(csv_dir),
            "--qlib_dir", str(qlib_dir),
            "--freq", "day",
            "--exclude_fields", "symbol,date",
            "--include_fields", "open,close,high,low,volume,factor,change",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise DependencyError(
            f"dump_bin failed for {sym}: rc={proc.returncode}",
            code="dump_failed",
            context={"stderr": proc.stderr[-500:]},
        )

    # Append/refresh entry in instruments/custom.txt with first/last dates from CSV.
    custom_inst = qlib_dir / "instruments" / "custom.txt"
    custom_inst.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if custom_inst.is_file():
        for ln in custom_inst.read_text(encoding="utf-8").splitlines():
            if ln.strip() and not ln.startswith(sym):
                lines.append(ln)
    csv_path = csv_dir / f"{sym}.csv"
    if csv_path.is_file():
        first_d = None
        last_d = None
        for ln in csv_path.read_text(encoding="utf-8").splitlines()[1:]:
            parts = ln.split(",")
            if len(parts) < 2:
                continue
            if first_d is None:
                first_d = parts[1]
            last_d = parts[1]
        if first_d and last_d:
            lines.append(f"{sym}\t{first_d}\t{last_d}")
    custom_inst.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return AddSymbolResponse(
        symbol=sym,
        fetched_rows=rows_appended,
        message=f"added & fetched {rows_appended} new rows",
    )


def _wait_and_update(job_id: str, proc: subprocess.Popen, log_path: Path) -> None:
    """Background-thread helper: wait for proc, update job entry, release the lock."""
    global _running_job_id
    try:
        rc = proc.wait()
    except Exception as e:
        with _running_state_lock:
            entry = _refresh_jobs.get(job_id)
            if entry is not None:
                entry["status"] = "failed"
                entry["finished_at"] = datetime.now(timezone.utc).isoformat()
                entry["error"] = str(e)
            _running_job_id = None
        try:
            _running_lock.release()
        except RuntimeError:
            pass
        return
    with _running_state_lock:
        entry = _refresh_jobs.get(job_id)
        if entry is not None:
            entry["status"] = "done" if rc == 0 else "failed"
            entry["finished_at"] = datetime.now(timezone.utc).isoformat()
            entry["returncode"] = rc
        _running_job_id = None
    try:
        _running_lock.release()
    except RuntimeError:
        pass


def get_active_refresh_job() -> dict | None:
    """Return the most recent non-terminal refresh job, or the most recent
    job overall if none is running. Lets the frontend recover progress
    after a page navigation / browser refresh.

    Shape: {"job_id": str, "status": "running"|"done"|"failed", "started_at": str}
           or None if no jobs have ever been started this process.
    """
    with _running_state_lock:
        if _running_job_id and _running_job_id in _refresh_jobs:
            entry = _refresh_jobs[_running_job_id]
            return {
                "job_id": entry["job_id"],
                "status": entry["status"],
                "started_at": entry["started_at"],
            }
        # No running job — return the most recently finished one so the UI
        # can render a "✓ done at …" badge after the user navigates back.
        if not _refresh_jobs:
            return None
        last = max(_refresh_jobs.values(), key=lambda e: e.get("started_at", ""))
        return {
            "job_id": last["job_id"],
            "status": last["status"],
            "started_at": last["started_at"],
        }


def start_refresh() -> RefreshResponse:
    global _running_job_id
    # Try to acquire the global running lock without blocking
    if not _running_lock.acquire(blocking=False):
        raise ConflictError(
            "refresh already in progress",
            code="refresh_in_progress",
            context={"running_job_id": _running_job_id},
        )
    try:
        job_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat()
        root = _worktree_root()
        script = root / "production" / "incremental_refresh.py"
        logs_dir = root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"api_refresh_{job_id}.log"

        log_fh = open(log_path, "wb")
        proc = subprocess.Popen(
            [_sys.executable, str(script)],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(root),
        )
        with _running_state_lock:
            _refresh_jobs[job_id] = {
                "job_id": job_id,
                "pid": proc.pid,
                "started_at": started_at,
                "finished_at": None,
                "status": "running",
                "log_path": str(log_path),
                "_proc": proc,
                "_log_fh": log_fh,
            }
            _running_job_id = job_id

        # Spawn a daemon thread to wait for the process; it will release the lock.
        t = threading.Thread(
            target=_wait_and_update,
            args=(job_id, proc, log_path),
            daemon=True,
        )
        t.start()

        return RefreshResponse(
            job_id=job_id,
            started_at=started_at,
            message="refresh started",
        )
    except Exception:
        # If we failed to spawn at all, release the lock.
        try:
            _running_lock.release()
        except RuntimeError:
            pass
        raise


def _tail_log(log_path: Path, n_lines: int = 50) -> str | None:
    if not log_path.is_file():
        return None
    try:
        # Read latest bytes to avoid loading huge logs
        with log_path.open("rb") as f:
            try:
                f.seek(0, 2)
                size = f.tell()
                read_size = min(size, 64 * 1024)
                f.seek(size - read_size)
                data = f.read()
            except Exception:
                f.seek(0)
                data = f.read()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception:
        return None


def _latest_progress(log_path: Path) -> ProgressInfo | None:
    """Parse the latest PROGRESS line from the log.

    The Python script emits lines like:
        PROGRESS {"phase":"fetch","current":42,"total":300,"message":"..."}

    Returns None if the log is missing, unreadable, or contains no parseable
    PROGRESS line.
    """
    if not log_path.is_file():
        return None
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 32 * 1024)
            f.seek(size - read_size)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        # Iterate lines from end; find the latest valid "PROGRESS {...}"
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line.startswith("PROGRESS "):
                continue
            try:
                payload = _json.loads(line[len("PROGRESS "):])
                return ProgressInfo(**payload)
            except Exception:
                continue
        return None
    except Exception:
        return None


def get_refresh_status(job_id: str) -> RefreshJobStatus:
    entry = _refresh_jobs.get(job_id)
    if entry is None:
        raise NotFoundError(
            "job not found",
            code="job_missing",
            context={"job_id": job_id},
        )
    log_path = Path(entry["log_path"])
    return RefreshJobStatus(
        job_id=entry["job_id"],
        status=entry["status"],
        started_at=entry["started_at"],
        finished_at=entry.get("finished_at"),
        log_tail=_tail_log(log_path),
        progress=_latest_progress(log_path),
    )
