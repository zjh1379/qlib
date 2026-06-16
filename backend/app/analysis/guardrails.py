"""Deterministic post-LLM guardrails for the AI analysis layer.

Pure functions, no I/O. Runs between LLM parse and persistence: the LLM
*proposes*, these rules *verify*. We never upgrade the model's view, only
ground it against the sources we actually fed in and keep it internally
consistent. Every intervention is recorded in `adjustments` for audit.
"""
from __future__ import annotations

from datetime import date, timedelta

from app.analysis.schemas import AnalysisResult
from app.analysis.sources import NewsItem, NoticeItem


def _norm(s: str) -> str:
    return "".join((s or "").split())


def _containment(a: str, b: str) -> float:
    """Fraction of the shorter string's distinct chars present in the longer."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    small, big = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return len(small & big) / len(small)


def _matches_any(source: str, titles: list[str], threshold: float = 0.7) -> bool:
    """A flag is grounded if its cited source is a substring of, or shares
    enough characters with, some title we actually fed the model. Tolerates
    LLM paraphrase of official titles without matching unrelated events."""
    s = _norm(source)
    if not s:
        return False
    for t in titles:
        nt = _norm(t)
        if not nt:
            continue
        if s in nt or nt in s:
            return True
        if len(s) >= 5 and _containment(s, nt) >= threshold:
            return True
    return False


def _parse_date(s: str | None) -> date | None:
    try:
        return date.fromisoformat((s or "").strip()[:10])
    except ValueError:
        return None


def _date_ok(source_date: str, as_of: date | None, lookback_days: int) -> bool:
    d = _parse_date(source_date)
    if d is None:
        return False
    if as_of is None:
        return True
    return as_of - timedelta(days=lookback_days) <= d <= as_of


def apply_guardrails(
    result: AnalysisResult, *,
    news: list[NewsItem], notices: list[NoticeItem],
    lookback_days: int = 60, as_of_date: str | None = None,
) -> tuple[AnalysisResult, list[str]]:
    """Verify `result` against the provided sources. Returns (result, adjustments)."""
    titles = [n.title for n in notices] + [n.title for n in news]
    as_of = _parse_date(as_of_date)
    adjustments: list[str] = []
    for flag in result.risk_flags:
        src_ok = _matches_any(flag.source, titles)
        date_ok = _date_ok(flag.source_date, as_of, lookback_days)
        flag.verified = src_ok and date_ok
        if not src_ok:
            adjustments.append(
                f"风险旗标来源未在所给新闻/公告中匹配,标记未核验:{flag.source}")
        elif not date_ok:
            adjustments.append(
                f"风险旗标来源日期无效或超出{lookback_days}天窗口,标记未核验:"
                f"{flag.source}({flag.source_date})")

    # Single-direction stance gate: only ever downgrade toward neutral, never up.
    verified_sev = {f.severity for f in result.risk_flags if f.verified}
    if result.stance == "caution" and not (verified_sev & {"high", "medium"}):
        adjustments.append("stance 由 caution 降为 neutral:无已核验的中高风险旗标")
        result.stance = "neutral"
    elif result.stance == "favorable" and "high" in verified_sev:
        adjustments.append("stance 由 favorable 降为 neutral:存在已核验的高风险旗标")
        result.stance = "neutral"

    return result, adjustments
