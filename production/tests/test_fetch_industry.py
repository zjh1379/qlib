import pandas as pd
from production.fetch_industry import parse_industry_rows, to_qlib_code


def test_to_qlib_code():
    assert to_qlib_code("sh.600000") == "SH600000"
    assert to_qlib_code("sz.000001") == "SZ000001"


def test_parse_industry_rows():
    # baostock query_stock_industry row format: [updateDate, code, code_name, industry, industryClassification]
    rows = [
        ["2024-01-01", "sh.600000", "浦发银行", "银行", "申万一级"],
        ["2024-01-01", "sz.300750", "宁德时代", "电池", "申万一级"],
        ["2024-01-01", "sz.000002", "万科A", "", "申万一级"],  # empty industry
    ]
    df = parse_industry_rows(rows)
    assert list(df.columns) == ["instrument", "industry"]
    m = dict(zip(df["instrument"], df["industry"]))
    assert m["SH600000"] == "银行"
    assert m["SZ300750"] == "电池"
    # empty industry filled with "UNKNOWN"
    assert m["SZ000002"] == "UNKNOWN"
