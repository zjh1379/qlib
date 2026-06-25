"""One-off: sample a training subprocess's RSS over time + tag phases, so we
know how much of the ~17GB peak is the Alpha360 data handler vs everything
else. Run alongside a real single-model train. Writes production/cache/mem_probe.csv."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True, help="training python.exe pid to watch")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--out", default=str(REPO_ROOT / "production" / "cache" / "mem_probe.csv"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    p = psutil.Process(args.pid)
    peak = 0.0
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "rss_gb", "children_rss_gb", "commit_pct"])
        while p.is_running():
            try:
                rss = p.memory_info().rss / 2**30
                ch = sum((c.memory_info().rss for c in p.children(recursive=True)), 0) / 2**30
                vm, sw = psutil.virtual_memory(), psutil.swap_memory()
                commit_pct = 100.0 * (vm.used + sw.used) / (vm.total + sw.total)
                peak = max(peak, rss + ch)
                w.writerow([time.strftime("%H:%M:%S"), f"{rss:.2f}", f"{ch:.2f}", f"{commit_pct:.1f}"])
                f.flush()
                time.sleep(args.interval)
            except psutil.NoSuchProcess:
                break
    print(f"PEAK total RSS = {peak:.2f} GB  (csv: {out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
