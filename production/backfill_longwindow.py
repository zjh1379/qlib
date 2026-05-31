# production/backfill_longwindow.py
"""Orchestrate the long-window backfill: LGBM (semi-annual) then ALSTM+TRA
(annual), sequentially, resumable, under safety_watchdog. Run in background.

PRE-FLIGHT (do manually before launching — see plan T5):
  1. Ensure pagefile >= 64GB.   2. Start safety_watchdog in another terminal:
     F:/Tools/Anaconda/envs/qlib/python.exe -m production.safety_watchdog
  3. Close heavy browser tabs.

Usage (background):
  F:/Tools/Anaconda/envs/qlib/python.exe -m production.backfill_longwindow \
     --start 2021-01-01 --end 2026-01-01 > logs/backfill_long.log 2>&1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PY = sys.executable
REPO = Path(__file__).resolve().parent.parent


def build_backfill_cmd(model: str, start: str, end: str, *, step_weeks: int,
                       test_weeks: int, train_years: int) -> list[str]:
    return [PY, "-m", "production.rolling_train", "backfill",
            "--start", start, "--end", end,
            "--only-models", model,
            "--step-weeks", str(step_weeks),
            "--test-weeks", str(test_weeks),
            "--train-years", str(train_years)]


def _run(cmd: list[str]) -> int:
    print(">>>", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--skip-neural", action="store_true",
                    help="LGBM baseline only (fast, safe)")
    a = ap.parse_args()

    # 1) LGBM semi-annual (light, fast). Resumable via run_once skip_if_exists.
    rc = _run(build_backfill_cmd("lgbm", a.start, a.end,
                                 step_weeks=26, test_weeks=26, train_years=3))
    if rc != 0:
        print("LGBM backfill returned", rc)
    if a.skip_neural:
        return 0
    # 2) Neural annual (heavy) — one model at a time, watchdog protects memory.
    for model in ("alstm", "tra"):
        rc = _run(build_backfill_cmd(model, a.start, a.end,
                                     step_weeks=52, test_weeks=52, train_years=3))
        if rc != 0:
            print(f"{model} backfill returned", rc, "(continuing; resumable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
