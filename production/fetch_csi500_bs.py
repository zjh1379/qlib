"""Fetch CSI500 + CSI300 stock names from baostock, merge into cn_names_cache.json.

The existing cache covers CSI300 only. After universe expanded to 800 (CSI300+CSI500),
many recommendations show blank names. This script extends the cache to cover all
A-share stocks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import baostock as bs
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = REPO_ROOT / "production" / "cn_names_cache.json"


def fetch_all_stock_names() -> dict[str, str]:
    """Pull every A-share stock's code+name. Returns {bare_code: name}, e.g. {'600519': '贵州茅台'}.

    Uses bs.query_stock_basic() which returns the FULL listing roster — much
    broader than CSI300/CSI500 alone.
    """
    bs.login()
    try:
        rs = bs.query_stock_basic()
        rows = []
        while (rs.error_code == "0") and rs.next():
            r = rs.get_row_data()  # [code, code_name, ipoDate, outDate, type, status]
            # type 1 = stock; we skip ETFs/indexes here (those come from etf_names.json)
            if len(r) >= 5 and r[4] == "1" and r[5] == "1":  # type=stock & status=listed
                bs_code = r[0]  # e.g. "sh.600519"
                bare = bs_code.split(".")[1]  # "600519"
                rows.append((bare, r[1]))
        return dict(rows)
    finally:
        bs.logout()


def main() -> int:
    print("Fetching all A-share stock names from baostock...", file=sys.stderr)
    fresh = fetch_all_stock_names()
    print(f"  Got {len(fresh)} stocks", file=sys.stderr)

    # Load existing cache + merge
    existing = {}
    if CACHE_PATH.exists():
        try:
            blob = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            existing = blob.get("map", {})
        except Exception as exc:
            print(f"  Warning: existing cache unreadable ({exc}), starting fresh", file=sys.stderr)

    # Merge: new data wins (refreshed names take precedence)
    merged = {**existing, **fresh}
    print(f"  Cache: existing={len(existing)} → merged={len(merged)} (+{len(merged)-len(existing)} new)", file=sys.stderr)

    # Write back, preserving the same JSON structure
    payload = {
        "built_at": pd.Timestamp.now().isoformat(),
        "source": "baostock query_stock_basic",
        "count": len(merged),
        "map": merged,
    }
    CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Wrote {CACHE_PATH} ({CACHE_PATH.stat().st_size // 1024} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
