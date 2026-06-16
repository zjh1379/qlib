"""Provider-agnostic LLM call for one pick. Sync (runs in a worker thread).

Supports OpenAI and DeepSeek (both via the OpenAI-compatible `openai` SDK, JSON
mode + client-side Pydantic validation) and Anthropic (native `messages.parse`).
Pick the provider with Settings.ai_provider.
"""
from __future__ import annotations

import logging
import os

from app.analysis.guardrails import apply_guardrails
from app.analysis.prompt import JSON_INSTRUCTION, SYSTEM, build_user_message
from app.analysis.schemas import AiAnalysis, AnalysisResult
from app.analysis.sources import NewsItem, NoticeItem

log = logging.getLogger(__name__)

# Per-provider default model when Settings.ai_model is blank.
_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "anthropic": "claude-sonnet-4-6",
}
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _provider(settings) -> str:
    return (settings.ai_provider or "openai").lower()


def _resolve_key(settings) -> str | None:
    """The active provider's API key (config first, then the conventional env var)."""
    p = _provider(settings)
    if p == "anthropic":
        return settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
    if p == "deepseek":
        return settings.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
    return settings.openai_api_key or os.getenv("OPENAI_API_KEY")


def make_client(settings):
    """Build the LLM client for the configured provider.

    Returns (kind, client, model) where kind is 'anthropic' or 'openai'
    ('openai' covers DeepSeek via base_url). model resolves to the per-provider
    default when Settings.ai_model is blank.
    """
    p = _provider(settings)
    model = settings.ai_model or _DEFAULT_MODEL.get(p, "gpt-4o-mini")
    if p == "anthropic":
        import anthropic
        return "anthropic", anthropic.Anthropic(api_key=settings.anthropic_api_key or None), model
    from openai import OpenAI
    if p == "deepseek":
        return "openai", OpenAI(
            api_key=settings.deepseek_api_key or None, base_url=_DEEPSEEK_BASE_URL,
        ), model
    return "openai", OpenAI(api_key=settings.openai_api_key or None), model


def analyze_one(
    kind: str, client, model: str, *, symbol: str, name: str,
    news: list[NewsItem], notices: list[NoticeItem], context: dict, as_of_date: str,
) -> AiAnalysis:
    """One LLM call -> AiAnalysis. Raises on API/parse error (caller decides retry)."""
    user = build_user_message(
        symbol=symbol, name=name, news=news, notices=notices, context=context)
    if kind == "anthropic":
        resp = client.messages.parse(
            model=model, max_tokens=1024, system=SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_format=AnalysisResult,
        )
        r: AnalysisResult = resp.parsed_output
    else:
        # OpenAI / DeepSeek: JSON mode + client-side Pydantic validation (portable
        # across both; DeepSeek doesn't support strict json_schema parse).
        resp = client.chat.completions.create(
            model=model, max_tokens=1024, temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM + "\n\n" + JSON_INSTRUCTION},
                {"role": "user", "content": user},
            ],
        )
        r = AnalysisResult.model_validate_json(resp.choices[0].message.content)
    # Deterministic guardrails: ground risk flags in the sources we fed in,
    # validate dates, keep stance internally consistent (LLM proposes, rules verify).
    r, adjustments = apply_guardrails(r, news=news, notices=notices, as_of_date=as_of_date)
    status = "ok" if (news or notices) else "partial"  # context-only = partial
    return AiAnalysis(
        interpretation=r.interpretation, risk_flags=r.risk_flags, stance=r.stance,
        model=model, as_of_date=as_of_date, status=status, adjustments=adjustments,
        news_count=len(news), notice_count=len(notices),
    )
