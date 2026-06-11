import pandas as pd
from app.analysis import sources


def test_to_ak_symbol_strips_exchange_prefix():
    assert sources.to_ak_symbol("SH600519") == "600519"
    assert sources.to_ak_symbol("SZ000001") == "000001"
    assert sources.to_ak_symbol("BJ830799") == "830799"
    assert sources.to_ak_symbol("600519") == "600519"


def test_fetch_news_normalizes_and_truncates(monkeypatch):
    fake = pd.DataFrame({
        "新闻标题": [f"t{i}" for i in range(30)],
        "发布时间": ["2026-06-10 09:00:00"] * 30,
        "文章来源": ["东方财富"] * 30,
        "新闻内容": ["..."] * 30,
    })
    monkeypatch.setattr(sources, "_ak_stock_news", lambda sym: fake)
    items = sources.fetch_news("SH600519", limit=15)
    assert len(items) == 15
    assert items[0].title == "t0"
    assert items[0].source == "东方财富"


def test_fetch_news_failsoft(monkeypatch):
    def boom(sym):
        raise RuntimeError("akshare down")
    monkeypatch.setattr(sources, "_ak_stock_news", boom)
    assert sources.fetch_news("SH600519") == []


def test_fetch_notices_normalizes_and_truncates(monkeypatch):
    fake = pd.DataFrame({
        "公告标题": [f"n{i}" for i in range(20)],
        "公告时间": ["2026-06-03"] * 20,
    })
    monkeypatch.setattr(sources, "_ak_disclosure", lambda sym, start, end: fake)
    items = sources.fetch_notices("SH600519", limit=10)
    assert len(items) == 10
    assert items[0].title == "n0"
    assert items[0].date == "2026-06-03"


def test_fetch_notices_failsoft(monkeypatch):
    def boom(sym, start, end):
        raise RuntimeError("cninfo down")
    monkeypatch.setattr(sources, "_ak_disclosure", boom)
    assert sources.fetch_notices("SH600519") == []
