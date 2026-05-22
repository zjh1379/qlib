"""Pure helpers for Tier 1 screener filters.

These are kept free of qlib / DB dependencies so they can be tested in isolation
and reused from both the service layer (apply filters server-side) and any
future scripts.
"""
from __future__ import annotations

import re

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
