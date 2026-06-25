"""Intraday entry-timing rules. Each rule returns a MULTIPLIER m relative to the
day's open (effective_entry = day_open * m), so it is 复权-invariant (same-day
ratio). Returns None when a BUY is not fillable (A-share limit-up). The simulator
multiplies m onto the daily ADJUSTED open."""
from __future__ import annotations
import pandas as pd


def bs_code(instrument: str) -> str:
    """SH600519 -> sh.600519 ; SZ000001 -> sz.000001 (baostock format)."""
    return f"{instrument[:2].lower()}.{instrument[2:]}"


def _limit_pct(instrument: str) -> float:
    code = instrument[2:]
    if instrument.startswith("BJ") or code.startswith(("43", "83", "87", "88", "92")):
        return 0.30                       # 北交所
    if code.startswith("688"):
        return 0.20                       # 科创板
    if code.startswith("30"):
        return 0.20                       # 创业板
    return 0.10                           # 主板


def limit_up_price(instrument: str, prev_close: float) -> float:
    return round(prev_close * (1 + _limit_pct(instrument)), 2)


def is_buy_fillable(day_bars: pd.DataFrame, prev_close: float, instrument: str) -> bool:
    """Not fillable if the stock never trades below its limit-up price (一字/封板涨停):
    a buyer can't get filled below the ceiling all day."""
    if day_bars is None or day_bars.empty:
        return False
    lu = limit_up_price(instrument, prev_close)
    return float(day_bars["low"].min()) < lu - 1e-9


def entry_multiplier(day_bars: pd.DataFrame, prev_close: float, instrument: str,
                     rule: str = "open", *, k: float = 0.01, g: float = 0.03,
                     first_n: int = 6) -> float | None:
    if day_bars is None or day_bars.empty:
        return None
    if not is_buy_fillable(day_bars, prev_close, instrument):
        return None
    # day open: prefer the first bar's own open, then its close (09:35 price ~ open);
    # baostock occasionally records the first bar's open as 0/NaN or emits sporadic
    # zero-price glitch bars, so fall back to the first later positive open/close.
    o = float(day_bars["open"].iloc[0])
    if not (o > 0):
        o = float(day_bars["close"].iloc[0])
    if not (o > 0):
        pos_open = day_bars["open"][day_bars["open"] > 0]
        o = float(pos_open.iloc[0]) if len(pos_open) else 0.0
    if not (o > 0):
        pos_close0 = day_bars["close"][day_bars["close"] > 0]
        o = float(pos_close0.iloc[0]) if len(pos_close0) else 0.0
    if not (o > 0):
        return None
    pos_close = day_bars["close"][day_bars["close"] > 0]
    close = float(pos_close.iloc[-1]) if len(pos_close) else o
    if rule == "open":
        price = o
    elif rule in ("vwap", "vwap_am"):
        bars = day_bars.iloc[:max(1, len(day_bars) // 2)] if rule == "vwap_am" else day_bars
        valid = bars[(bars["volume"] > 0) & (bars["amount"] > 0)]
        vol = float(valid["volume"].sum())
        price = float(valid["amount"].sum()) / vol if vol > 0 else o
    elif rule == "low_band":
        pos_low = day_bars["low"][day_bars["low"] > 0]
        day_low = float(pos_low.min()) if len(pos_low) else o
        band = o * (1 - k)
        price = band if day_low <= band else close
    elif rule == "gap_cond":
        gap = o / prev_close - 1 if prev_close > 0 else 0.0
        if gap >= g:
            return None                   # don't chase a gap-up
        price = o
    elif rule == "first30_low":
        first = day_bars["low"].iloc[:first_n]
        pos_first = first[first > 0]
        price = float(pos_first.min()) if len(pos_first) else o
    elif rule == "am30_vwap":
        # realistic "bought sometime in the first 30 min" = VWAP of the first
        # first_n bars (6 = 30 min). Non-look-ahead (achievable by spreading buys).
        bars = day_bars.iloc[:first_n]
        valid = bars[(bars["volume"] > 0) & (bars["amount"] > 0)]
        vol = float(valid["volume"].sum())
        price = float(valid["amount"].sum()) / vol if vol > 0 else o
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return price / o if o > 0 else None
