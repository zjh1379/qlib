from __future__ import annotations

from app.analysis.sources import NewsItem, NoticeItem

SYSTEM = (
    "你是A股短线研究助理。模型用日频均值反转(超跌反弹)选出候选股,你的任务是结合"
    "近期新闻/公告做定性二次解读,辅助人最后拍板。\n"
    "硬规则:\n"
    "1. 风险旗标只能基于我提供的新闻/公告,每条必须引用来源标题与日期,不准凭模型先验推断。\n"
    "2. 没有命中实质性利空就返回空的 risk_flags,不要硬凑。\n"
    "3. interpretation 用一句中文说明为什么这只票现在值得关注(结合反转信号+近期消息),"
    "客观、不喊单。\n"
    "4. stance 是参考倾向(favorable/neutral/caution),不是交易信号,不影响量化排名。\n"
    "5. 优先关注公告与实质性事件,弱化纯行情复述类新闻。"
)


def _fmt_pct(v) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "NA"


def build_user_message(
    symbol: str, name: str,
    news: list[NewsItem], notices: list[NoticeItem], context: dict,
) -> str:
    lines = [f"股票:{name or ''} ({symbol})"]
    ctx_bits = [
        f"量化分数={context.get('score_today')}",
        f"近5日涨跌={_fmt_pct(context.get('pct_change_5d'))}",
        f"板块={context.get('board', 'NA')}",
        f"ST={'是' if context.get('is_st') else '否'}",
    ]
    lines.append("量化上下文:" + " | ".join(ctx_bits))

    if notices:
        lines.append("近期公告:")
        lines += [f"- [{n.date}] {n.title}" for n in notices]
    else:
        lines.append("近期公告:暂无可用数据")

    if news:
        lines.append("近期新闻:")
        lines += [f"- [{n.date}] {n.title}（{n.source}）" for n in news]
    else:
        lines.append("近期新闻:暂无")

    lines.append("\n请输出结构化结果(interpretation / risk_flags / stance)。")
    return "\n".join(lines)
