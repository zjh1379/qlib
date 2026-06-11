from app.analysis.llm import analyze_one
from app.analysis.schemas import AnalysisResult, RiskFlag
from app.analysis.sources import NewsItem


class _FakeParsed:
    def __init__(self, result): self.parsed_output = result


class _FakeMessages:
    def __init__(self, result): self._r = result; self.calls = []
    def parse(self, **kw): self.calls.append(kw); return _FakeParsed(self._r)


class _FakeClient:
    def __init__(self, result): self.messages = _FakeMessages(result)


def test_analyze_one_maps_to_ai_analysis():
    result = AnalysisResult(
        interpretation="超跌+业绩预喜", stance="favorable",
        risk_flags=[RiskFlag(type="解禁", severity="medium", reason="下周解禁",
                             source="解禁公告", source_date="2026-06-09")],
    )
    client = _FakeClient(result)
    out = analyze_one(client, symbol="SH600519", name="贵州茅台",
                      news=[NewsItem(title="业绩预喜")], notices=[],
                      context={"score_today": 0.9}, model="claude-opus-4-8",
                      as_of_date="2026-06-10")
    assert out.interpretation == "超跌+业绩预喜"
    assert out.stance == "favorable"
    assert out.risk_flags[0].type == "解禁"
    assert out.model == "claude-opus-4-8"
    assert out.as_of_date == "2026-06-10"
    assert out.status == "ok"
    assert client.messages.calls[0]["output_format"] is AnalysisResult
    assert client.messages.calls[0]["model"] == "claude-opus-4-8"


def test_analyze_one_status_partial_when_no_sources():
    result = AnalysisResult(interpretation="仅凭分数", stance="neutral", risk_flags=[])
    out = analyze_one(_FakeClient(result), symbol="SH600519", name="x",
                      news=[], notices=[], context={"score_today": 0.5},
                      model="claude-opus-4-8", as_of_date="2026-06-10")
    assert out.status == "partial"
