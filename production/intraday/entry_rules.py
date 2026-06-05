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
    o = float(day_bars["open"].iloc[0])
    if not (o > 0):
        # baostock data glitch: first 5min bar's open is occasionally 0/NaN even
        # when the stock traded all day -> use the first bar's close as day-open proxy.
        o = float(day_bars["close"].iloc[0])
    if not (o > 0):
        return None
    close = float(day_bars["close"].iloc[-1])
    if rule == "open":
        price = o
    elif rule == "vwap":
        vol = float(day_bars["volume"].sum())
        price = float(day_bars["amount"].sum()) / vol if vol > 0 else o
    elif rule == "vwap_am":
        am = day_bars[day_bars.index < len(day_bars) // 2] if len(day_bars) else day_bars
        vol = float(am["volume"].sum())
        price = float(am["amount"].sum()) / vol if vol > 0 else o
    elif rule == "low_band":
        band = o * (1 - k)
        price = band if float(day_bars["low"].min()) <= band else close
    elif rule == "gap_cond":
        gap = o / prev_close - 1 if prev_close > 0 else 0.0
        if gap >= g:
            return None                   # don't chase a gap-up
        price = o
    elif rule == "first30_low":
        price = float(day_bars["low"].iloc[:first_n].min())
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return price / o
