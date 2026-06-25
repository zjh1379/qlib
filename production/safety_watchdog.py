"""System-wide commit-charge watchdog (Windows OOM prevention).

This is the LAST LINE OF DEFENSE against the hard system freezes documented
in Event 2004 (`Windows successfully diagnosed a low virtual memory
condition`) which lead to Event 41 (`Kernel-Power 41 unexpected shutdown`)
that required a hard reboot.

Failure mode it prevents:
  1. Training subprocess (e.g. ALSTM-Alpha360) grows to 17+ GB virtual
     memory due to handler accumulation
  2. Combined with other processes (Chrome, Claude, backend), total
     commit charge approaches the 60 GB Windows limit (RAM 32 + pagefile 28)
  3. Once commit exhausts, the entire OS freezes — UI repaint stops,
     keyboard becomes unresponsive, only hard reboot recovers

Watchdog logic:
  - Every 2 seconds, sample psutil.virtual_memory() + swap_memory()
  - When commit > WARN_PCT, log warning + identify heaviest training-related
    python.exe by RSS
  - When commit > KILL_PCT, send SIGTERM to the heaviest training subprocess
    (preferring rolling_train / run_split / daily_inference children),
    then SIGKILL after a grace period

Whitelist: never kill uvicorn (backend), vite, claude.exe, chrome.exe, or
explorer.exe. Only target ML training subprocesses we own.

Usage:
  # Foreground (interactive)
  python -m production.safety_watchdog

  # Background (run before kicking off a long training)
  start /b python -m production.safety_watchdog > logs/watchdog.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import psutil

log = logging.getLogger("safety_watchdog")

# Defaults — tuned for 32GB RAM + 28GB pagefile = 60GB commit limit.
WARN_PCT = 80.0    # log a warning above this
KILL_PCT = 92.0    # kill heaviest training subprocess above this
GRACE_SECONDS = 5  # SIGTERM -> wait -> SIGKILL window

# Process names + cmdline substrings we are allowed to kill. Anything else
# (uvicorn, vite, chrome, explorer, the user's editor, etc.) is left alone.
KILLABLE_TOKENS = (
    "rolling_train",
    "production.run_split",
    "production.daily_inference",
    "production.incremental_refresh",
    "production.backfill_calibration",
    "production.train_alstm",
    "production.train_tra",
    "production.walk_forward",
)


def is_killable_cmd(cmd: str) -> bool:
    """True iff the command line matches a training process we are allowed to
    kill. Infra (uvicorn/vite/chrome/claude/explorer) never matches."""
    return any(tok in cmd for tok in KILLABLE_TOKENS)


def _commit_pct() -> tuple[float, float, float]:
    """Returns (commit_pct, used_gb, total_gb)."""
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    used = vm.used + sw.used
    total = vm.total + sw.total
    pct = 100.0 * used / total if total else 0.0
    return pct, used / 2**30, total / 2**30


def decide_action(
    pct: float,
    used_gb: float,
    total_gb: float,
    *,
    warn_pct: float,
    kill_pct: float,
    floor_gb: float,
) -> str:
    """Pure decision: 'kill' | 'warn' | 'ok'.

    Kills when commit pct crosses kill_pct OR when *absolute* free commit
    (total - used) drops below floor_gb. The floor matters because on a small
    commit ceiling a percentage threshold reacts too late — by the time pct is
    high the few remaining GB vanish within one poll interval.
    """
    free_gb = total_gb - used_gb
    if pct >= kill_pct or free_gb < floor_gb:
        return "kill"
    if pct >= warn_pct:
        return "warn"
    return "ok"


def record_kill(path, record: dict) -> None:
    """Append one JSON line describing a watchdog kill. Fail-soft; path=None
    skips. The backend tails this file to surface 'killed by OOM guard'."""
    if path is None:
        return
    try:
        import time as _t
        rec = {"ts": _t.strftime("%Y-%m-%dT%H:%M:%S"), **record}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _find_heaviest_killable() -> Optional[psutil.Process]:
    """Walk all processes, return the heaviest one whose cmdline matches
    KILLABLE_TOKENS. Returns None if no killable training process found.
    """
    best: Optional[psutil.Process] = None
    best_rss = 0
    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            if p.info.get("name") != "python.exe":
                continue
            cmd = " ".join(p.info.get("cmdline") or [])
            if not is_killable_cmd(cmd):
                continue
            rss = p.info["memory_info"].rss
            if rss > best_rss:
                best_rss = rss
                best = p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return best


def _kill_with_grace(p: psutil.Process, reason: str, kills_path=None) -> None:
    try:
        rss_gb = p.memory_info().rss / 2**30
        cmd = " ".join(p.cmdline())[:200]
        log.error("KILLING pid=%d rss=%.1fGB cmd=%s — %s", p.pid, rss_gb, cmd, reason)
        record_kill(kills_path, {"pid": p.pid, "rss_gb": round(rss_gb, 2), "cmd": cmd, "reason": reason})
        for ch in p.children(recursive=True):
            try:
                ch.terminate()
            except Exception:
                pass
        p.terminate()
        gone, alive = psutil.wait_procs([p], timeout=GRACE_SECONDS)
        for survivor in alive:
            log.error("force-killing pid=%d (didn't terminate gracefully)", survivor.pid)
            try:
                survivor.kill()
            except Exception:
                pass
    except Exception as exc:
        log.exception("kill failed pid=%d: %s", p.pid, exc)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--warn-pct", type=float, default=WARN_PCT,
                        help=f"commit pct to log warning (default {WARN_PCT})")
    parser.add_argument("--kill-pct", type=float, default=KILL_PCT,
                        help=f"commit pct to kill heaviest training proc (default {KILL_PCT})")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="poll interval seconds (default 2.0)")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--floor-gb", type=float, default=4.0,
                        help="kill heaviest training proc if free commit < this many GB (default 4)")
    parser.add_argument("--kills-path", default=None,
                        help="append JSON kill records here (e.g. logs/watchdog_kills.jsonl)")
    args = parser.parse_args()

    _setup_logging(args.log_level)
    log.info("watchdog started warn=%.1f%% kill=%.1f%% interval=%.1fs",
             args.warn_pct, args.kill_pct, args.interval)

    consecutive_kill_attempts = 0
    last_warn_at = 0.0

    def _handle_sigint(_sig, _frm):
        log.info("watchdog terminating on signal")
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except Exception:
        pass

    while True:
        try:
            pct, used_gb, total_gb = _commit_pct()
            now = time.time()
            action = decide_action(
                pct, used_gb, total_gb,
                warn_pct=args.warn_pct, kill_pct=args.kill_pct,
                floor_gb=getattr(args, "floor_gb", 4.0),
            )
            if action == "kill":
                target = _find_heaviest_killable()
                if target:
                    _kill_with_grace(
                        target,
                        reason=f"commit {pct:.1f}% / free {total_gb-used_gb:.1f}GB crossed limits",
                        kills_path=getattr(args, "kills_path", None),
                    )
                    consecutive_kill_attempts += 1
                    if consecutive_kill_attempts > 3:
                        log.error("too many kill attempts — pausing 30s to let OS recover")
                        time.sleep(30)
                        consecutive_kill_attempts = 0
                else:
                    if now - last_warn_at > 30:
                        log.warning("commit %.1f%% CRITICAL but no killable training proc", pct)
                        last_warn_at = now
            elif action == "warn":
                consecutive_kill_attempts = 0
                if now - last_warn_at > 30:
                    log.warning("commit %.1f%% (%.1f / %.1f GB) approaching limit", pct, used_gb, total_gb)
                    last_warn_at = now
            else:
                consecutive_kill_attempts = 0
            time.sleep(args.interval)
        except Exception as exc:
            log.exception("watchdog loop error: %s", exc)
            time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main() or 0)
