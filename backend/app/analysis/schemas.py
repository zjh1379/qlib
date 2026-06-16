from typing import Literal
from pydantic import BaseModel, Field

Severity = Literal["high", "medium", "low"]
Stance = Literal["favorable", "neutral", "caution"]


class RiskFlag(BaseModel):
    type: str                 # 立案/退市/商誉/解禁/业绩预警/诉讼/其他
    severity: Severity
    reason: str               # short, grounded in the cited source
    source: str               # the news/announcement title it came from
    source_date: str          # ISO date of the source
    verified: bool = True      # set by guardrails: source matched a provided item


class AnalysisResult(BaseModel):
    """Exactly what Claude returns (structured output). We add model/date/status."""
    interpretation: str
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    stance: Stance


class AiAnalysis(BaseModel):
    """Served packet attached to a ScreenItem."""
    interpretation: str
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    stance: Stance = "neutral"
    model: str = ""
    as_of_date: str = ""
    status: str = "ok"        # ok | partial | failed
    adjustments: list[str] = Field(default_factory=list)  # guardrail interventions (audit)
    news_count: int = 0       # provenance: how many news items fed the model
    notice_count: int = 0     # provenance: how many announcements fed the model


class AnalysisJob(BaseModel):
    job_id: str
    status: str               # running | done | failed
    started_at: str
    finished_at: str | None = None
    analyzed: int | None = None       # number of picks analyzed on success
    as_of_date: str | None = None
    error: str | None = None
    reason: str | None = None         # data_refresh | manual_ui


class AnalysisStatus(BaseModel):
    last_run_at: str | None = None
    last_success_at: str | None = None
    last_error: str | None = None
    is_running: bool = False


class TriggerResponse(BaseModel):
    status: str               # started | already_running | disabled
    job_id: str | None = None
