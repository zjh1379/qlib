"""akshare news + announcement fetch for the AI analysis layer.

akshare is heavy and only needed here — import lazily. All fetches are
best-effort: any failure returns [] so a single bad symbol never breaks the
batch. Called from a worker thread (sync).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from pydantic import BaseModel

log = logging.getLogger(__name__)

_PREFIXES = ("SH", "SZ", "BJ")


class NewsItem(BaseModel):
    title: str
    date: str = ""
    source: str = ""


class NoticeItem(BaseModel):
    title: str
    date: str = ""
    kind: str = ""


def to_ak_symbol(symbol: str) -> str:
    """SH600519 -> 600519 (akshare wants the bare 6-digit code)."""
    if len(symbol) > 2 and symbol[:2] in _PREFIXES:
        return symbol[2:]
    return symbol


def _ak_stock_news(ak_symbol: str):
    import akshare as ak
    return ak.stock_news_em(symbol=ak_symbol)


def _ak_disclosure(ak_symbol: str, start_date: str, end_date: str):
    import akshare as ak
    return ak.stock_zh_a_disclosure_report_cninfo(
        symbol=ak_symbol, market="沪深京", start_date=start_date, end_date=end_date,
    )


def fetch_news(symbol: str, limit: int = 15) -> list[NewsItem]:
    try:
        df = _ak_stock_news(to_ak_symbol(symbol))
    except Exception as exc:
        log.warning("news_fetch_failed symbol=%s: %s", symbol, exc)
        return []
    out: list[NewsItem] = []
    for r in df.head(limit).to_dict("records"):
        title = str(r.get("新闻标题", "") or "").strip()
        if not title:
            continue
        out.append(NewsItem(
            title=title,
            date=str(r.get("发布时间", "") or ""),
            source=str(r.get("文章来源", "") or ""),
        ))
    return out


def fetch_notices(symbol: str, limit: int = 15, lookback_days: int = 60) -> list[NoticeItem]:
    end = date.today().strftime("%Y%m%d")
    start = (date.today() - timedelta(days=lookback_days)).strftime("%Y%m%d")
    try:
        df = _ak_disclosure(to_ak_symbol(symbol), start, end)
    except Exception as exc:
        log.warning("notice_fetch_failed symbol=%s: %s", symbol, exc)
        return []
    out: list[NoticeItem] = []
    for r in df.head(limit).to_dict("records"):
        title = str(r.get("公告标题", "") or "").strip()
        if not title:
            continue
        out.append(NoticeItem(title=title, date=str(r.get("公告时间", "") or "")))
    return out
