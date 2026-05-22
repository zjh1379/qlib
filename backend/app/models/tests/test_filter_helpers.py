import pytest

from app.models.utils import is_st_name, parse_board


@pytest.mark.parametrize(
    "symbol, expected",
    [
        ("SH600519", "main"),   # 沪市主板 (60xxxx)
        ("SZ000001", "main"),   # 深市主板 (00xxxx)
        ("SH601318", "main"),   # 沪市主板 (60xxxx)
        ("SZ300750", "gem"),    # 创业板 (30xxxx)
        ("SH688981", "star"),   # 科创板 (688xxx)
        ("SH689009", "star"),   # 科创板 (689xxx, though 688 is the common prefix)
        ("BJ430047", "bj"),     # 北交所 (BJ exchange)
        ("BJ831010", "bj"),
        ("SH510300", "etf"),    # 沪市 ETF (51xxxx)
        ("SZ159995", "etf"),    # 深市 ETF (15xxxx/16xxxx/17xxxx)
        ("SH588000", "etf"),    # 科创 ETF (588xxx — by convention ETFs even though 58x)
    ],
)
def test_parse_board(symbol, expected):
    assert parse_board(symbol) == expected


def test_parse_board_unknown_returns_other():
    assert parse_board("XX999999") == "other"
    assert parse_board("") == "other"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ST康美", True),
        ("*ST康美", True),
        ("ST 康美", True),       # trailing/leading whitespace + space variant
        ("ST*康美", True),       # variant ordering
        ("贵州茅台", False),
        ("茅台", False),
        ("", False),
        ("STAR Holdings", False),  # contains "STA" but not ST as a token boundary
    ],
)
def test_is_st_name(name, expected):
    assert is_st_name(name) == expected
