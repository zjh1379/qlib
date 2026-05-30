"""Fetch SW industry classification from baostock; save instrument->industry map.

Usage:
  F:/Tools/Anaconda/envs/qlib/python.exe -m production.fetch_industry \
    --out production/cache/industry_map.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def to_qlib_code(bs_code: str) -> str:
    """'sh.600000' -> 'SH600000'."""
    market, num = bs_code.split(".")
    return f"{market.upper()}{num}"


def parse_industry_rows(rows: list[list[str]]) -> pd.DataFrame:
    """Parse baostock query_stock_industry rows into instrument->industry.
    Row layout: [updateDate, code, code_name, industry, industryClassification]."""
    recs = []
    for r in rows:
        if len(r) < 4:
            continue
        code = to_qlib_code(r[1])
        industry = (r[3] or "").strip() or "UNKNOWN"
        recs.append({"instrument": code, "industry": industry})
    return pd.DataFrame(recs, columns=["instrument", "industry"])


def fetch_industry_map() -> pd.DataFrame:
    import baostock as bs
    lg = bs.login()
    try:
        rs = bs.query_stock_industry()
        if rs.error_code != "0":
            raise RuntimeError(f"query_stock_industry failed: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
    finally:
        bs.logout()
    return parse_industry_rows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch baostock SW industry map.")
    ap.add_argument("--out", default="production/cache/industry_map.parquet")
    args = ap.parse_args()
    df = fetch_industry_map()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} rows={len(df)} industries={df['industry'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
