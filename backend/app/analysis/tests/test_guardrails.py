from app.analysis.guardrails import apply_guardrails
from app.analysis.schemas import AnalysisResult, RiskFlag
from app.analysis.sources import NewsItem, NoticeItem


def _flag(source, *, severity="medium", source_date="2026-06-09", type="其他"):
    return RiskFlag(type=type, severity=severity, reason="r",
                    source=source, source_date=source_date)


def _result(stance, flags):
    return AnalysisResult(interpretation="x", stance=stance, risk_flags=flags)


# ---- R1: source grounding (anti-hallucination veto) -----------------------
def test_unmatched_source_is_marked_unverified():
    notices = [NoticeItem(title="股东减持解禁公告", date="2026-06-09")]
    result = _result("caution", [
        _flag("股东减持解禁公告"),                 # grounded in a provided notice
        _flag("证监会立案调查", severity="high"),  # never fed in -> hallucinated
    ])
    out, adjustments = apply_guardrails(
        result, news=[], notices=notices, as_of_date="2026-06-10")
    by_source = {f.source: f for f in out.risk_flags}
    assert by_source["股东减持解禁公告"].verified is True
    assert by_source["证监会立案调查"].verified is False
    assert any("证监会立案调查" in a for a in adjustments)


# ---- R2: date validity (forced recent, parseable date) --------------------
def test_source_date_outside_lookback_window_is_unverified():
    notices = [NoticeItem(title="重大诉讼公告", date="2026-06-08")]
    result = _result("caution", [_flag("重大诉讼公告", source_date="2026-01-01")])
    out, adj = apply_guardrails(result, news=[], notices=notices,
                                lookback_days=60, as_of_date="2026-06-10")
    assert out.risk_flags[0].verified is False
    assert any("日期" in a for a in adj)


def test_unparseable_source_date_is_unverified():
    notices = [NoticeItem(title="重大诉讼公告", date="2026-06-08")]
    result = _result("caution", [_flag("重大诉讼公告", source_date="近期")])
    out, _adj = apply_guardrails(result, news=[], notices=notices,
                                 as_of_date="2026-06-10")
    assert out.risk_flags[0].verified is False


# ---- R3: stance <-> flags single-direction gate (only ever downgrades) ----
def test_caution_without_verified_material_flag_downgraded_to_neutral():
    notices = [NoticeItem(title="日常经营公告", date="2026-06-09")]
    result = _result("caution", [_flag("证监会立案", severity="high")])  # not in sources
    out, adj = apply_guardrails(result, news=[], notices=notices, as_of_date="2026-06-10")
    assert out.stance == "neutral"
    assert any("stance" in a for a in adj)


def test_favorable_with_verified_high_flag_downgraded_to_neutral():
    notices = [NoticeItem(title="公司被证监会立案调查的公告", date="2026-06-09")]
    result = _result("favorable", [_flag("公司被证监会立案调查的公告", severity="high")])
    out, adj = apply_guardrails(result, news=[], notices=notices, as_of_date="2026-06-10")
    assert out.risk_flags[0].verified is True
    assert out.stance == "neutral"
    assert any("stance" in a for a in adj)


def test_favorable_stays_with_only_verified_medium_flag():
    notices = [NoticeItem(title="股东解禁公告", date="2026-06-09")]
    result = _result("favorable", [_flag("股东解禁公告", severity="medium")])
    out, _adj = apply_guardrails(result, news=[], notices=notices, as_of_date="2026-06-10")
    assert out.stance == "favorable"


# ---- R1b: fuzzy match tolerates LLM paraphrase of the official title ------
def test_paraphrased_source_still_counts_as_grounded():
    notices = [NoticeItem(title="关于收到中国证券监督管理委员会立案告知书的公告",
                          date="2026-06-09")]
    result = _result("caution", [_flag("收到证监会立案告知书", severity="high")])
    out, _adj = apply_guardrails(result, news=[], notices=notices, as_of_date="2026-06-10")
    assert out.risk_flags[0].verified is True


def test_unrelated_source_does_not_fuzzy_match():
    notices = [NoticeItem(title="关于收到中国证券监督管理委员会立案告知书的公告",
                          date="2026-06-09")]
    result = _result("caution", [_flag("拟收购新能源资产暨重大资产重组", severity="high")])
    out, _adj = apply_guardrails(result, news=[], notices=notices, as_of_date="2026-06-10")
    assert out.risk_flags[0].verified is False
