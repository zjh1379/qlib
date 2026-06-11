"""Claude structured-output call for one pick. Sync (runs in a worker thread)."""
from __future__ import annotations

import logging

from app.analysis.prompt import SYSTEM, build_user_message
from app.analysis.schemas import AiAnalysis, AnalysisResult
from app.analysis.sources import NewsItem, NoticeItem

log = logging.getLogger(__name__)


def make_client(api_key: str | None):
    import anthropic
    # api_key="" -> None lets the SDK fall back to the ANTHROPIC_API_KEY env var.
    return anthropic.Anthropic(api_key=api_key or None)


def analyze_one(
    client, *, symbol: str, name: str,
    news: list[NewsItem], notices: list[NoticeItem], context: dict,
    model: str, as_of_date: str,
) -> AiAnalysis:
    """One Claude call. Raises on API error (caller decides retry/failed)."""
    resp = client.messages.parse(
        model=model,
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_user_message(
            symbol=symbol, name=name, news=news, notices=notices, context=context)}],
        output_format=AnalysisResult,
    )
    r: AnalysisResult = resp.parsed_output
    status = "ok" if (news or notices) else "partial"  # context-only = partial
    return AiAnalysis(
        interpretation=r.interpretation,
        risk_flags=r.risk_flags,
        stance=r.stance,
        model=model,
        as_of_date=as_of_date,
        status=status,
    )
