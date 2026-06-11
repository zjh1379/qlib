from app.analysis.prompt import SYSTEM, build_user_message
from app.analysis.sources import NewsItem, NoticeItem


def test_system_has_anti_hallucination_rules():
    assert "只" in SYSTEM and "来源" in SYSTEM        # must-cite-source rule
    assert "不是交易信号" in SYSTEM                    # stance disclaimer


def test_user_message_includes_titles_and_context():
    msg = build_user_message(
        symbol="SH600519", name="贵州茅台",
        news=[NewsItem(title="茅台发布业绩预告", date="2026-06-10", source="东财")],
        notices=[NoticeItem(title="股东解禁公告", date="2026-06-09")],
        context={"score_today": 0.9, "pct_change_5d": -0.08, "board": "main", "is_st": False},
    )
    assert "贵州茅台" in msg
    assert "茅台发布业绩预告" in msg
    assert "股东解禁公告" in msg
    assert "-8" in msg or "-0.08" in msg            # recent move surfaced


def test_user_message_handles_empty_sources():
    msg = build_user_message("SH600519", "贵州茅台", [], [], {"score_today": 0.5})
    assert "无" in msg or "暂无" in msg               # graceful empty-state
