import numpy as np, pandas as pd, pytest
from production.intraday.entry_rules import (
    bs_code, limit_up_price, is_buy_fillable, entry_multiplier)


def _day(opens, highs, lows, closes, vols):  # one day's 5min bars
    n = len(opens)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": vols, "amount": np.array(closes) * np.array(vols)})


def test_bs_code():
    assert bs_code("SH600519") == "sh.600519"
    assert bs_code("SZ000001") == "sz.000001"


def test_limit_up_price_by_board():
    assert limit_up_price("SH600000", 10.0) == pytest.approx(11.0)   # main 10%
    assert limit_up_price("SZ300750", 10.0) == pytest.approx(12.0)   # ChiNext 20%
    assert limit_up_price("SH688981", 10.0) == pytest.approx(12.0)   # STAR 20%
    assert limit_up_price("BJ830799", 10.0) == pytest.approx(13.0)   # BJ 30%


def test_open_multiplier_is_one():
    d = _day([10, 10.1], [10.2, 10.2], [9.9, 10.0], [10.1, 10.1], [100, 100])
    assert entry_multiplier(d, prev_close=9.8, instrument="SH600000", rule="open") == pytest.approx(1.0)


def test_low_band_filled_when_touched():
    # open 10, dips to 9.8 (=-2%): low_band(0.01)=9.9 touched -> entry 9.9 -> mult 0.99
    d = _day([10, 9.8], [10.0, 9.9], [10.0, 9.8], [10.0, 9.85], [100, 100])
    m = entry_multiplier(d, 9.9, "SH600000", rule="low_band", k=0.01)
    assert m == pytest.approx(0.99)


def test_low_band_missed_uses_close():
    # never dips to band -> fill at day close
    d = _day([10, 10.1], [10.2, 10.2], [10.0, 10.05], [10.1, 10.15], [100, 100])
    m = entry_multiplier(d, 9.9, "SH600000", rule="low_band", k=0.02)  # band 9.8 not touched
    assert m == pytest.approx(10.15 / 10.0)


def test_vwap_multiplier():
    d = _day([10, 10], [10, 10], [10, 10], [10, 12], [100, 100])  # closes 10,12 vol 100,100
    # vwap = (10*100+12*100)/200 = 11 ; open=10 -> mult 1.1
    assert entry_multiplier(d, 9.9, "SH600000", rule="vwap") == pytest.approx(1.1)


def test_gap_cond_skips_big_gap_up():
    # open 10, prev_close 9.0 -> gap +11% >= g(3%) -> not chasing -> None
    d = _day([10, 10.1], [10.2, 10.2], [9.95, 10.0], [10.1, 10.1], [100, 100])
    assert entry_multiplier(d, 9.0, "SH600000", rule="gap_cond", g=0.03) is None


def test_gap_cond_buys_on_low_open():
    d = _day([10, 10.1], [10.2, 10.2], [9.95, 10.0], [10.1, 10.1], [100, 100])
    assert entry_multiplier(d, 10.5, "SH600000", rule="gap_cond", g=0.03) == pytest.approx(1.0)  # gap down -> open


def test_first30_low_uses_min_first6_bars():
    opens = [10, 10, 10, 10, 10, 10, 10]; lows = [10, 9.9, 9.8, 9.95, 10, 10, 10]
    d = _day(opens, [11] * 7, lows, [10] * 7, [100] * 7)
    # min low over first 6 bars = 9.8 -> mult 0.98
    assert entry_multiplier(d, 9.9, "SH600000", rule="first30_low") == pytest.approx(0.98)


def test_one_zi_limit_up_not_fillable():
    # 一字涨停: whole day at limit (open=high=low=close=11.0=prev*1.1) -> buy unfillable
    d = _day([11.0, 11.0], [11.0, 11.0], [11.0, 11.0], [11.0, 11.0], [100, 100])
    assert is_buy_fillable(d, prev_close=10.0, instrument="SH600000") is False
    assert entry_multiplier(d, 10.0, "SH600000", rule="open") is None


def test_vwap_ignores_zero_volume_glitch_bar():
    # middle bar is a zero glitch (vol 0, amount 0) -> excluded from vwap
    d = _day([10, 10, 10], [10, 10, 10], [10, 0, 10], [10, 0, 12], [100, 0, 100])
    # amount = close*vol = [1000, 0, 1200]; valid bars (vol>0): vwap=(1000+1200)/200=11; open=10 -> 1.1
    assert entry_multiplier(d, 9.8, "SH600000", rule="vwap") == pytest.approx(1.1)


def test_first30_low_ignores_zero_low_glitch_bar():
    # a glitched 0.0 low inside first 6 bars must NOT become the entry price
    opens = [10, 10, 10, 10, 10, 10]; lows = [10, 9.9, 0.0, 9.8, 10, 10]
    d = _day(opens, [11] * 6, lows, [10] * 6, [100] * 6)
    # valid lows over first 6 = {10,9.9,9.8,10,10} -> min 9.8 -> mult 0.98 (not 0.0)
    assert entry_multiplier(d, 9.9, "SH600000", rule="first30_low") == pytest.approx(0.98)


def test_first_bar_zero_open_falls_back_to_close():
    # baostock data glitch: first 5min bar open=0 -> day-open proxy = first bar close=10.0
    d = _day([0, 10.1], [10.2, 10.2], [9.9, 10.0], [10.0, 10.1], [100, 100])
    # vwap amount = closes*vols = [1000,1010]; vwap=2010/200=10.05; day-open=10.0 -> mult 1.005
    assert entry_multiplier(d, 9.8, "SH600000", rule="vwap") == pytest.approx(10.05 / 10.0)
    # open rule still 1.0 (price == day-open proxy)
    assert entry_multiplier(d, 9.8, "SH600000", rule="open") == pytest.approx(1.0)


def test_parse_baostock_5min_types():
    from production.intraday.fetch_5min import parse_baostock_5min
    raw = pd.DataFrame({
        "time": ["20241202093500000", "20241202094000000"],
        "open": ["10.0", "10.1"], "high": ["10.2", "10.2"], "low": ["9.9", "10.0"],
        "close": ["10.1", "10.15"], "volume": ["100", "200"], "amount": ["1010", "2030"]})
    out = parse_baostock_5min(raw)
    assert str(out["datetime"].iloc[0]) == "2024-12-02 09:35:00"
    assert out["open"].dtype.kind == "f" and out["volume"].iloc[1] == 200.0
    assert list(out.columns)[:1] == ["datetime"]
