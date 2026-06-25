"""Start/stop the system commit-charge watchdog (production.safety_watchdog)
as a managed child process for the FastAPI lifespan. The watchdog is the
last-line OOM defense; before this it was a standalone script users forgot
to run. Fail-soft: a failed watchdog start must never block app startup."""
from __future__ import annotations

import subprocess
from pathlib import Path

from app.core.logging import get_logger

_log = get_logger("watchdog")


def watchdog_cmd(python_path: str, repo_root: Path, *, floor_gb: float,
                 kill_pct: float, kills_path: Path) -> list[str]:
    """Pure: build the watchdog subprocess argv."""
    return [
        python_path, "-m", "production.safety_watchdog",
        "--floor-gb", str(floor_gb),
        "--kill-pct", str(kill_pct),
        "--kills-path", str(kills_path),
    ]


def start_watchdog(python_path: str, repo_root: Path, *, floor_gb: float = 4.0,
                   kill_pct: float = 92.0) -> subprocess.Popen | None:
    """Launch the watchdog with its own log file. Returns the Popen or None on
    failure (never raises)."""
    try:
        logs = repo_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        kills_path = logs / "watchdog_kills.jsonl"
        cmd = watchdog_cmd(python_path, repo_root, floor_gb=floor_gb,
                           kill_pct=kill_pct, kills_path=kills_path)
        logf = (logs / "watchdog.log").open("ab")
        proc = subprocess.Popen(cmd, cwd=str(repo_root), stdout=logf, stderr=subprocess.STDOUT)
        _log.info("watchdog_started", pid=proc.pid, cmd=" ".join(cmd))
        return proc
    except Exception as exc:
        _log.warning("watchdog_start_failed", error=str(exc))
        return None


def stop_watchdog(proc: subprocess.Popen | None) -> None:
    """Terminate the watchdog child. Fail-soft."""
    if proc is None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        _log.info("watchdog_stopped")
    except Exception as exc:
        _log.warning("watchdog_stop_failed", error=str(exc))
