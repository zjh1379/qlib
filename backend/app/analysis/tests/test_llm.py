from types import SimpleNamespace

from app.analysis.llm import analyze_one, make_client
from app.analysis.schemas import AnalysisResult, RiskFlag
from app.analysis.sources import NewsItem, NoticeItem

_RESULT = AnalysisResult(
    interpretation="超跌+业绩预喜", stance="favorable",
    risk_flags=[RiskFlag(type="解禁", severity="medium", reason="下周解禁",
                         source="解禁公告", source_date="2026-06-09")],
)


# ---- anthropic path (native messages.parse) -------------------------------
class _FakeParsed:
    def __init__(self, result): self.parsed_output = result

class _FakeMessages:
    def __init__(self, result): self._r = result; self.calls = []
    def parse(self, **kw): self.calls.append(kw); return _FakeParsed(self._r)

class _FakeAnthropic:
    def __init__(self, result): self.messages = _FakeMessages(result)


def test_analyze_one_anthropic_path():
    client = _FakeAnthropic(_RESULT)
    out = analyze_one("anthropic", client, "claude-sonnet-4-6",
                      symbol="SH600519", name="贵州茅台",
                      news=[NewsItem(title="业绩预喜")], notices=[],
                      context={"score_today": 0.9}, as_of_date="2026-06-10")
    assert out.interpretation == "超跌+业绩预喜"
    assert out.risk_flags[0].type == "解禁"
    assert out.model == "claude-sonnet-4-6"
    assert out.status == "ok"
    assert client.messages.calls[0]["output_format"] is AnalysisResult


# ---- openai / deepseek path (JSON mode + client-side validation) -----------
class _Msg:
    def __init__(self, content): self.message = SimpleNamespace(content=content)

class _Completions:
    def __init__(self, content): self._c = content; self.calls = []
    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(choices=[_Msg(self._c)])

class _FakeOpenAI:
    def __init__(self, content): self.chat = SimpleNamespace(completions=_Completions(content))


def test_analyze_one_openai_path_parses_json():
    client = _FakeOpenAI(_RESULT.model_dump_json())
    out = analyze_one("openai", client, "gpt-4o-mini",
                      symbol="SH600519", name="贵州茅台",
                      news=[NewsItem(title="业绩预喜")], notices=[],
                      context={"score_today": 0.9}, as_of_date="2026-06-10")
    assert out.interpretation == "超跌+业绩预喜"
    assert out.risk_flags[0].type == "解禁"
    assert out.model == "gpt-4o-mini"
    assert out.status == "ok"
    assert client.chat.completions.calls[0]["response_format"] == {"type": "json_object"}
    assert client.chat.completions.calls[0]["model"] == "gpt-4o-mini"


def test_analyze_one_runs_guardrails_on_hallucinated_flag():
    bad = AnalysisResult(
        interpretation="x", stance="caution",
        risk_flags=[RiskFlag(type="立案", severity="high", reason="被立案",
                             source="查无此公告", source_date="2026-06-09")])
    client = _FakeOpenAI(bad.model_dump_json())
    out = analyze_one("openai", client, "gpt-4o-mini", symbol="SH600519", name="x",
                      news=[NewsItem(title="业绩预喜")], notices=[],
                      context={"score_today": 0.5}, as_of_date="2026-06-10")
    assert out.risk_flags[0].verified is False   # source not among provided items
    assert out.adjustments                       # intervention recorded for audit


def test_analyze_one_reports_source_counts():
    client = _FakeOpenAI(_RESULT.model_dump_json())
    out = analyze_one("openai", client, "gpt-4o-mini", symbol="SH600519", name="x",
                      news=[NewsItem(title="a"), NewsItem(title="b")],
                      notices=[NoticeItem(title="c")],
                      context={"score_today": 0.5}, as_of_date="2026-06-10")
    assert out.news_count == 2
    assert out.notice_count == 1


def test_analyze_one_status_partial_when_no_sources():
    client = _FakeOpenAI(
        AnalysisResult(interpretation="仅凭分数", stance="neutral", risk_flags=[]).model_dump_json())
    out = analyze_one("openai", client, "gpt-4o-mini",
                      symbol="SH600519", name="x", news=[], notices=[],
                      context={"score_today": 0.5}, as_of_date="2026-06-10")
    assert out.status == "partial"


# ---- provider selection ---------------------------------------------------
def _settings(**kw):
    base = dict(ai_provider="openai", ai_model="", openai_api_key="",
                deepseek_api_key="", anthropic_api_key="")
    base.update(kw)
    return SimpleNamespace(**base)


def test_make_client_selects_provider_and_default_model():
    kind, _c, model = make_client(_settings(ai_provider="openai", openai_api_key="x"))
    assert (kind, model) == ("openai", "gpt-4o-mini")

    kind, c, model = make_client(_settings(ai_provider="deepseek", deepseek_api_key="x"))
    assert (kind, model) == ("openai", "deepseek-chat")
    assert "deepseek.com" in str(c.base_url)

    kind, _c, model = make_client(_settings(ai_provider="anthropic", anthropic_api_key="x"))
    assert (kind, model) == ("anthropic", "claude-sonnet-4-6")

    # explicit ai_model overrides the per-provider default
    _kind, _c, model = make_client(_settings(ai_provider="openai", openai_api_key="x",
                                             ai_model="gpt-4.1-mini"))
    assert model == "gpt-4.1-mini"
