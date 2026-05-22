"""Pure helpers for Tier 1 screener filters.

These are kept free of qlib / DB dependencies so they can be tested in isolation
and reused from both the service layer (apply filters server-side) and any
future scripts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Code-segment to board mapping per CSRC market structure:
#   60xxxx  / 90xxxx  -> 沪市主板 (Shanghai Main)
#   00xxxx           -> 深市主板 (Shenzhen Main)
#   30xxxx           -> 创业板 (ChiNext / GEM)
#   688xxx / 689xxx  -> 科创板 (STAR Market)
#   430xxx / 8xxxxx  -> 北交所 (Beijing Exchange) -- "BJ" prefix
#   51xxxx / 56xxxx / 58xxxx (excl. 688/689) -> 沪市 ETF
#   15xxxx / 16xxxx / 17xxxx                 -> 深市 ETF
def parse_board(symbol: str) -> str:
    """Classify a qlib-format symbol into 'main' | 'gem' | 'star' | 'bj' | 'etf' | 'other'."""
    if not symbol or len(symbol) < 8:
        return "other"
    prefix = symbol[:2].upper()
    code = symbol[2:]
    if prefix == "BJ":
        return "bj"
    if prefix == "SH":
        if code.startswith(("688", "689")):
            return "star"
        if code.startswith(("60", "90")):
            return "main"
        if code.startswith(("51", "56", "58")):
            return "etf"
    elif prefix == "SZ":
        if code.startswith("30"):
            return "gem"
        if code.startswith(("00",)):
            return "main"
        if code.startswith(("15", "16", "17")):
            return "etf"
    return "other"


# A股 ST patterns: "ST<name>", "*ST<name>", "ST <name>", or with the asterisk
# placed mid-name in some legacy exchanges. We detect the ST token at the start
# of the name (the canonical placement), allowing for optional leading * and
# whitespace.
_ST_PATTERN = re.compile(r"^\s*\*?\s*ST[\s\*]", re.IGNORECASE)


def is_st_name(name: str) -> bool:
    """Return True iff the company name has an ST / *ST risk marker at the start."""
    if not name:
        return False
    # Anchor to start; require ST to be followed by whitespace or asterisk;
    # bare 'ST' or 'ST<chinese>' goes through the fallback branch below.
    if _ST_PATTERN.match(name):
        return True
    # Also handle the exact-prefix "ST<chinese>" without a separator — Chinese
    # characters effectively act as a token boundary. Normalize to uppercase
    # to stay consistent with the case-insensitive regex above.
    stripped = name.lstrip().upper()
    if stripped.startswith(("ST", "*ST")):
        rest = stripped[3:] if stripped.startswith("*ST") else stripped[2:]
        if rest and not rest[0].isascii():
            return True
    return False


@dataclass
class Tier1FilterSpec:
    pct_change_n: int = 5
    min_pct_change: float | None = None
    max_pct_change: float | None = None
    min_amplitude: float | None = None
    max_amplitude: float | None = None
    min_vol_ratio: float | None = None
    max_vol_ratio: float | None = None
    new_high_n: int = 0  # 0 = off
    boards: set[str] | None = None  # None = no board filter; set means OR within boards
    exclude_st: bool = True


def _passes_range(value: float | None, lo: float | None, hi: float | None) -> bool:
    """Inclusive range check that treats either bound as 'unbounded' when None.
    A None value (e.g. metric not available) fails any non-None bound."""
    if lo is None and hi is None:
        return True
    if value is None:
        return False
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def apply_tier1_filters(
    rows: list[dict],
    spec: Tier1FilterSpec,
) -> list[dict]:
    """Apply Tier 1 filters with AND semantics. Each row must carry the metric
    fields produced by qlib_adapter.get_filter_metrics + a 'symbol' + 'name' + 'board'.

    Returns the rows that pass every filter, in input order.
    """
    out: list[dict] = []
    pct_key = f"pct_change_{spec.pct_change_n}d"
    for r in rows:
        if not _passes_range(r.get(pct_key), spec.min_pct_change, spec.max_pct_change):
            continue
        if not _passes_range(r.get("amplitude"), spec.min_amplitude, spec.max_amplitude):
            continue
        if not _passes_range(r.get("vol_ratio"), spec.min_vol_ratio, spec.max_vol_ratio):
            continue
        if spec.new_high_n != 0:
            key = f"is_new_high_{spec.new_high_n}d"
            if not r.get(key, False):
                continue
        if spec.boards is not None and len(spec.boards) > 0:
            if r.get("board") not in spec.boards:
                continue
        if spec.exclude_st and r.get("is_st", False):
            continue
        out.append(r)
    return out
