from app.analysis.schemas import (
    RiskFlag, AiAnalysis, AnalysisResult, AnalysisJob, AnalysisStatus, TriggerResponse,
)


def test_ai_analysis_defaults_and_flags():
    a = AiAnalysis(interpretation="超跌反弹候选", stance="favorable",
                   model="claude-opus-4-8", as_of_date="2026-06-10", status="ok",
                   risk_flags=[RiskFlag(type="立案", severity="high", reason="被证监会立案",
                                        source="某公告", source_date="2026-06-09")])
    assert a.risk_flags[0].severity == "high"
    assert a.status == "ok"


def test_analysis_result_is_llm_output_shape():
    r = AnalysisResult(interpretation="x", stance="neutral", risk_flags=[])
    assert r.stance == "neutral"
    assert not hasattr(r, "status")


def test_trigger_response_disabled():
    assert TriggerResponse(status="disabled").job_id is None
    j = AnalysisJob(job_id="abc", status="running", started_at="t")
    assert j.analyzed is None
    assert AnalysisStatus().is_running is False
