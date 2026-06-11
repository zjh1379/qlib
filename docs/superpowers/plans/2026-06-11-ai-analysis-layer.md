# AI 分析层 (解读 + 风险旗标) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For each day's top-N qlib picks, an LLM reads recent akshare news + announcements and produces a one-line interpretation plus risk flags, served alongside the picks — pure decision support, never altering qlib rankings.

**Architecture:** A new backend vertical slice `backend/app/analysis/` owns akshare fetching, the Claude call, SQLite persistence, an in-memory job tracker (mirroring `inference/service.py`), and REST endpoints. `production/daily_inference.py` triggers a localhost-only refresh after it appends predictions; the backend background job derives top-N from `models.service.candidates()`, analyzes each pick, and upserts into an `ai_analysis` table. The `/api/models/candidates` route LEFT-JOINs the stored analysis onto each `ScreenItem` at request time (no qlib-cache pollution). Frontend renders a risk-flag badge + expandable note on the Picks page and tracks the job via the existing `useActiveJobs` hook.

**Tech Stack:** FastAPI + async SQLAlchemy (aiosqlite) + Alembic; `anthropic` Python SDK (`messages.parse` structured outputs, model `claude-opus-4-8`); akshare (sync, lazy-imported, threadpool); React + Vite + react-query.

**Spec:** `docs/superpowers/specs/2026-06-11-ai-analysis-layer-design.md`

**Conventions discovered (match these):**
- Run backend tests from the MAIN repo's `backend/`: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest <path> -v`.
- ORM style = classic `Column(...)` + `__tablename__` + `__table_args__` + `server_default=func.current_timestamp()` (see `app/portfolio/orm.py`).
- Tables are created via **Alembic** migrations (`backend/alembic/versions/000{1,2,3}_*.py`); tests build schema with `Base.metadata.create_all` after importing the ORM module.
- Job tracker pattern (module-level `OrderedDict` + `threading.Lock` + daemon thread + `get_active_job`/`get_status`/`get_job`) = `app/inference/service.py`.
- Localhost-only internal endpoint pattern = `app/inference/router.py` `internal_router`.
- Model-ID default is `claude-opus-4-8` (Anthropic guidance: do not auto-downgrade for cost; expose Sonnet as a config lever).

---

## Task 0: Step-0 spike — akshare + Claude feasibility gate (BLOCKING)

Not TDD — a throwaway verification script. **If this fails, stop and report; do not build on an interface that doesn't deliver.**

**Files:**
- Create (throwaway): `production/research/_spike_ai_analysis.py`

- [ ] **Step 1: Install the Anthropic SDK in the qlib env**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pip install "anthropic>=0.50"`
Expected: installs cleanly. Note the installed version.

- [ ] **Step 2: Write the spike script**

```python
# production/research/_spike_ai_analysis.py
"""Throwaway Step-0 gate: confirm akshare news/notice shape + one real Claude parse."""
import os, json
import akshare as ak

SYM = "600519"  # Kweichow Moutai — guaranteed to have news

print("=== stock_news_em ===")
news = ak.stock_news_em(symbol=SYM)
print("cols:", list(news.columns))
print("rows:", len(news))
print(news.head(3).to_dict("records"))

print("\n=== announcements (try candidates) ===")
for fn, kwargs in [
    ("stock_notice_report", {"symbol": "全部", "date": "20260610"}),
    ("stock_zh_a_disclosure_report_cninfo",
     {"symbol": SYM, "market": "沪深京", "start_date": "20260501", "end_date": "20260611"}),
]:
    try:
        df = getattr(ak, fn)(**kwargs)
        print(f"{fn} OK cols={list(df.columns)} rows={len(df)}")
        print(df.head(2).to_dict("records"))
    except Exception as e:
        print(f"{fn} FAILED: {e}")

print("\n=== Claude structured parse ===")
import anthropic
from pydantic import BaseModel
from typing import Literal

class Flag(BaseModel):
    type: str; severity: Literal["high","medium","low"]; reason: str; source: str; source_date: str
class Out(BaseModel):
    interpretation: str; risk_flags: list[Flag]; stance: Literal["favorable","neutral","caution"]

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
titles = "\n".join(f"- {r.get('新闻标题','')}" for r in news.head(10).to_dict("records"))
resp = client.messages.parse(
    model="claude-opus-4-8", max_tokens=1024,
    system="你是A股研究助理。只依据提供的新闻/公告判断,风险旗标必须引用来源标题+日期,不准凭空推断。stance不是交易信号。",
    messages=[{"role":"user","content":f"股票 {SYM} 贵州茅台。近期新闻:\n{titles}\n\n给出一句话解读+风险旗标。"}],
    output_format=Out,
)
print(json.dumps(resp.parsed_output.model_dump(), ensure_ascii=False, indent=2))
```

- [ ] **Step 3: Run the spike (needs a real key)**

Run: `$env:ANTHROPIC_API_KEY="sk-ant-..."; cd /e/Projects/qlib; F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._spike_ai_analysis`
Expected: prints `stock_news_em` columns (note the exact Chinese column names for title/time/source/content), at least one working announcement interface (note its name + columns), and a valid parsed JSON with `interpretation`/`risk_flags`/`stance`.

- [ ] **Step 4: Record findings + gate decision**

Write the confirmed interface details into the spec's Step-0 section (edit `docs/superpowers/specs/2026-06-11-ai-analysis-layer-design.md`, Step 0):
- `stock_news_em` exact columns used for title/datetime/source.
- The working announcement interface name + columns (or "per-stock announcements NOT reliably available → v1 ships interpretation + news-only flags").
- Anthropic SDK version installed.

Gate: news works + parse works → proceed. Announcements unavailable → proceed with news-only flags (note it). akshare or parse fails hard → **stop and report**.

- [ ] **Step 5: Delete the spike + commit deps**

```bash
rm production/research/_spike_ai_analysis.py
# add anthropic to backend deps
```
Edit `backend/pyproject.toml` dependencies array: add `"anthropic>=0.50",`.

```bash
git add backend/pyproject.toml docs/superpowers/specs/2026-06-11-ai-analysis-layer-design.md
git commit -m "chore(ai-analysis): Step-0 spike findings + anthropic dep"
```

> Task 4's code is pre-filled with the **expected** `stock_news_em` column names (`新闻标题` / `发布时间` / `文章来源`). If Task 0 Step 4 recorded different names, update those three string literals (and the test fixture) to match. Everything else is fully specified.

---

## Task 1: Config knobs for the AI layer

**Files:**
- Modify: `backend/app/core/config.py`
- Test: `backend/app/core/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/app/core/tests/test_config.py`:

```python
def test_ai_analysis_settings_defaults_and_env(monkeypatch):
    from app.core.config import Settings
    s = Settings()
    assert s.ai_model == "claude-opus-4-8"   # do NOT auto-downgrade for cost
    assert s.ai_analysis_top_n == 10
    assert s.ai_analysis_enabled is False
    assert s.anthropic_api_key == ""

    monkeypatch.setenv("QLIB_COMPANION_ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("QLIB_COMPANION_AI_ANALYSIS_ENABLED", "true")
    monkeypatch.setenv("QLIB_COMPANION_AI_MODEL", "claude-sonnet-4-6")
    s2 = Settings()
    assert s2.anthropic_api_key == "sk-test"
    assert s2.ai_analysis_enabled is True
    assert s2.ai_model == "claude-sonnet-4-6"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/core/tests/test_config.py::test_ai_analysis_settings_defaults_and_env -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'ai_model'`.

- [ ] **Step 3: Add the settings**

In `backend/app/core/config.py`, inside `class Settings`, after the `retrain_python_path` line:

```python
    # AI analysis layer (解读 + 风险旗标)
    anthropic_api_key: str = ""           # QLIB_COMPANION_ANTHROPIC_API_KEY (env only, not committed)
    ai_model: str = "claude-opus-4-8"     # cost lever: set to claude-sonnet-4-6 to cut ~40%
    ai_analysis_top_n: int = 10           # analyze the top-N picks per run
    ai_analysis_enabled: bool = False     # off until a key is set
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/core/tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/config.py backend/app/core/tests/test_config.py
git commit -m "feat(ai-analysis): add config knobs (model/top_n/enabled/key)"
```

---

## Task 2: Schemas (served, LLM output, job)

**Files:**
- Create: `backend/app/analysis/__init__.py` (empty)
- Create: `backend/app/analysis/schemas.py`
- Create: `backend/app/analysis/tests/__init__.py` (empty)
- Test: `backend/app/analysis/tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/app/analysis/tests/test_schemas.py
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
    # AnalysisResult is what Claude returns (no model/as_of_date/status — we add those)
    r = AnalysisResult(interpretation="x", stance="neutral", risk_flags=[])
    assert r.stance == "neutral"
    assert not hasattr(r, "status")


def test_trigger_response_disabled():
    assert TriggerResponse(status="disabled").job_id is None
    j = AnalysisJob(job_id="abc", status="running", started_at="t")
    assert j.analyzed is None
    assert AnalysisStatus().is_running is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.analysis'`.

- [ ] **Step 3: Create the package + schemas**

Create empty `backend/app/analysis/__init__.py` and `backend/app/analysis/tests/__init__.py`. Then:

```python
# backend/app/analysis/schemas.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_schemas.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/analysis/__init__.py backend/app/analysis/schemas.py backend/app/analysis/tests/
git commit -m "feat(ai-analysis): schemas (RiskFlag/AiAnalysis/AnalysisResult/job)"
```

---

## Task 3: ORM table + Alembic migration

**Files:**
- Create: `backend/app/analysis/orm.py`
- Create: `backend/alembic/versions/0004_add_ai_analysis_table.py`
- Test: `backend/app/analysis/tests/test_orm.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/app/analysis/tests/test_orm.py
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.db import Base
from app.analysis.orm import AiAnalysisORM


@pytest.mark.asyncio
async def test_ai_analysis_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        s.add(AiAnalysisORM(symbol="SH600519", as_of_date="2026-06-10",
                            interpretation="x", risk_flags_json="[]",
                            stance="neutral", model="claude-opus-4-8", status="ok"))
        await s.commit()
        row = await s.get(AiAnalysisORM, {"symbol": "SH600519", "as_of_date": "2026-06-10"})
        assert row.interpretation == "x"
        assert row.created_at is not None
    await engine.dispose()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_orm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.analysis.orm'`.

- [ ] **Step 3: Write the ORM model**

```python
# backend/app/analysis/orm.py
from sqlalchemy import Column, DateTime, String
from sqlalchemy.sql import func

from app.core.db import Base


class AiAnalysisORM(Base):
    __tablename__ = "ai_analysis"

    symbol = Column(String, primary_key=True)
    as_of_date = Column(String, primary_key=True)   # ISO date the picks are as-of
    interpretation = Column(String, nullable=False, default="")
    risk_flags_json = Column(String, nullable=False, default="[]")  # JSON list[RiskFlag]
    stance = Column(String, nullable=False, default="neutral")
    model = Column(String, nullable=False, default="")
    status = Column(String, nullable=False, default="ok")  # ok | partial | failed
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_orm.py -v`
Expected: PASS.

- [ ] **Step 5: Write the Alembic migration**

First confirm the current head: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m alembic heads` — note the revision id (expected `0003`). Set `down_revision` to it.

```python
# backend/alembic/versions/0004_add_ai_analysis_table.py
"""add ai_analysis table

Revision ID: 0004
Revises: 0003
"""
from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"   # <- set to the id printed by `alembic heads`
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "ai_analysis",
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("as_of_date", sa.String(), nullable=False),
        sa.Column("interpretation", sa.String(), nullable=False, server_default=""),
        sa.Column("risk_flags_json", sa.String(), nullable=False, server_default="[]"),
        sa.Column("stance", sa.String(), nullable=False, server_default="neutral"),
        sa.Column("model", sa.String(), nullable=False, server_default=""),
        sa.Column("status", sa.String(), nullable=False, server_default="ok"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
        sa.PrimaryKeyConstraint("symbol", "as_of_date"),
    )


def downgrade():
    op.drop_table("ai_analysis")
```

- [ ] **Step 6: Apply the migration to the real app.db (MAIN repo)**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m alembic upgrade head`
Expected: `Running upgrade 0003 -> 0004, add ai_analysis table`.

- [ ] **Step 7: Commit**

```bash
git add backend/app/analysis/orm.py backend/app/analysis/tests/test_orm.py backend/alembic/versions/0004_add_ai_analysis_table.py
git commit -m "feat(ai-analysis): ai_analysis ORM + alembic migration 0004"
```

---

## Task 4: akshare data sources (news + announcements)

**Files:**
- Create: `backend/app/analysis/sources.py`
- Test: `backend/app/analysis/tests/test_sources.py`

> The code uses the expected `stock_news_em` column names `新闻标题` / `发布时间` / `文章来源`. If Task 0 recorded different names, update those literals + the test fixture. If announcements were unavailable in the spike, keep `fetch_notices` returning `[]` (a stub) and note it in the spec.

- [ ] **Step 1: Write the failing test**

```python
# backend/app/analysis/tests/test_sources.py
import pandas as pd
from app.analysis import sources


def test_to_ak_symbol_strips_exchange_prefix():
    assert sources.to_ak_symbol("SH600519") == "600519"
    assert sources.to_ak_symbol("SZ000001") == "000001"
    assert sources.to_ak_symbol("BJ830799") == "830799"
    assert sources.to_ak_symbol("600519") == "600519"


def test_fetch_news_normalizes_and_truncates(monkeypatch):
    fake = pd.DataFrame({
        "新闻标题": [f"t{i}" for i in range(30)],
        "发布时间": ["2026-06-10 09:00:00"] * 30,
        "文章来源": ["东方财富"] * 30,
    })
    monkeypatch.setattr(sources, "_ak_stock_news", lambda sym: fake)
    items = sources.fetch_news("SH600519", limit=15)
    assert len(items) == 15
    assert items[0].title == "t0"
    assert items[0].source == "东方财富"


def test_fetch_news_failsoft_on_error(monkeypatch):
    def boom(sym): raise RuntimeError("akshare down")
    monkeypatch.setattr(sources, "_ak_stock_news", boom)
    assert sources.fetch_news("SH600519") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_sources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.analysis.sources'`.

- [ ] **Step 3: Write the sources module**

```python
# backend/app/analysis/sources.py
"""akshare news + announcement fetch for the AI analysis layer.

akshare is heavy and only needed here — import lazily. All fetches are
best-effort: any failure returns [] so a single bad symbol never breaks the
batch. Called from a worker thread (sync), wrap with run_in_threadpool if used
from async code.
"""
from __future__ import annotations

import logging
from pydantic import BaseModel

log = logging.getLogger(__name__)

_PREFIXES = ("SH", "SZ", "BJ")


class NewsItem(BaseModel):
    title: str
    date: str = ""
    source: str = ""


class NoticeItem(BaseModel):
    title: str
    date: str = ""
    kind: str = ""


def to_ak_symbol(symbol: str) -> str:
    """SH600519 -> 600519 (akshare wants the bare 6-digit code)."""
    if len(symbol) > 2 and symbol[:2] in _PREFIXES:
        return symbol[2:]
    return symbol


def _ak_stock_news(ak_symbol: str):
    import akshare as ak
    return ak.stock_news_em(symbol=ak_symbol)


def fetch_news(symbol: str, limit: int = 15) -> list[NewsItem]:
    try:
        df = _ak_stock_news(to_ak_symbol(symbol))
    except Exception as exc:
        log.warning("news_fetch_failed symbol=%s: %s", symbol, exc)
        return []
    out: list[NewsItem] = []
    for r in df.head(limit).to_dict("records"):
        # Expected stock_news_em columns; confirm/adjust per Task 0 findings.
        title = str(r.get("新闻标题", "") or "").strip()
        if not title:
            continue
        out.append(NewsItem(
            title=title,
            date=str(r.get("发布时间", "") or ""),
            source=str(r.get("文章来源", "") or ""),
        ))
    return out


def fetch_notices(symbol: str, limit: int = 15) -> list[NoticeItem]:
    """Per-stock announcements. If Task-0 found no reliable per-stock interface,
    leave this as a stub returning [] and note it in the spec."""
    return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_sources.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/analysis/sources.py backend/app/analysis/tests/test_sources.py
git commit -m "feat(ai-analysis): akshare news/notice sources (lazy, fail-soft)"
```

---

## Task 5: Prompt builder (with anti-hallucination rules)

**Files:**
- Create: `backend/app/analysis/prompt.py`
- Test: `backend/app/analysis/tests/test_prompt.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/app/analysis/tests/test_prompt.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.analysis.prompt'`.

- [ ] **Step 3: Write the prompt builder**

```python
# backend/app/analysis/prompt.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_prompt.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/analysis/prompt.py backend/app/analysis/tests/test_prompt.py
git commit -m "feat(ai-analysis): grounded prompt builder + anti-hallucination system"
```

---

## Task 6: LLM call (Claude structured output) + mapping to AiAnalysis

**Files:**
- Create: `backend/app/analysis/llm.py`
- Test: `backend/app/analysis/tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/app/analysis/tests/test_llm.py
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
    # output_format must be the structured class
    assert client.messages.calls[0]["output_format"] is AnalysisResult
    assert client.messages.calls[0]["model"] == "claude-opus-4-8"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.analysis.llm'`.

- [ ] **Step 3: Write the LLM module**

```python
# backend/app/analysis/llm.py
"""Claude structured-output call for one pick. Sync (runs in a worker thread)."""
from __future__ import annotations

import logging

from app.analysis.prompt import SYSTEM, build_user_message
from app.analysis.schemas import AiAnalysis, AnalysisResult
from app.analysis.sources import NewsItem, NoticeItem

log = logging.getLogger(__name__)


def make_client(api_key: str | None):
    import anthropic
    # api_key="" -> None lets the SDK fall back to ANTHROPIC_API_KEY env var.
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_llm.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/analysis/llm.py backend/app/analysis/tests/test_llm.py
git commit -m "feat(ai-analysis): Claude structured-output call -> AiAnalysis"
```

---

## Task 7: Persistence (sqlite upsert from worker) + async serving read

**Files:**
- Create: `backend/app/analysis/store.py`
- Test: `backend/app/analysis/tests/test_store.py`

> The worker thread can't use the async session, so writes go through a plain `sqlite3` connection to the same `app.db`. Serving reads use the async ORM. Same table (created by migration 0004).

- [ ] **Step 1: Write the failing test**

```python
# backend/app/analysis/tests/test_store.py
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.db import Base
from app.analysis import store
from app.analysis.orm import AiAnalysisORM  # noqa: F401 (register table)
from app.analysis.schemas import AiAnalysis, RiskFlag


def _mk(symbol, status="ok"):
    return AiAnalysis(interpretation=f"note-{symbol}", stance="neutral",
                      model="claude-opus-4-8", as_of_date="2026-06-10", status=status,
                      risk_flags=[RiskFlag(type="解禁", severity="low", reason="r",
                                           source="s", source_date="2026-06-09")])


@pytest.mark.asyncio
async def test_upsert_then_fetch(tmp_path):
    db = tmp_path / "app.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # sync write (as the worker does)
    store.upsert_many(str(db), [_mk("SH600519"), _mk("SZ000001")])
    store.upsert_many(str(db), [_mk("SH600519")])  # idempotent overwrite

    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        got = await store.fetch_analyses(s, ["SH600519", "SZ000001", "SZ999999"], "2026-06-10")
    assert set(got) == {"SH600519", "SZ000001"}
    assert got["SH600519"].risk_flags[0].type == "解禁"
    assert got["SH600519"].as_of_date == "2026-06-10"
    await engine.dispose()


@pytest.mark.asyncio
async def test_fetch_excludes_failed(tmp_path):
    db = tmp_path / "app.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    store.upsert_many(str(db), [_mk("SH600519", status="failed")])
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        got = await store.fetch_analyses(s, ["SH600519"], "2026-06-10")
    assert got == {}
    await engine.dispose()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.analysis.store'`.

- [ ] **Step 3: Write the store module**

```python
# backend/app/analysis/store.py
"""ai_analysis persistence. Writes are sync sqlite3 (worker thread); reads are
async ORM (serving path). Same table, created by alembic migration 0004."""
from __future__ import annotations

import json
import logging
import sqlite3

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.orm import AiAnalysisORM
from app.analysis.schemas import AiAnalysis, RiskFlag

log = logging.getLogger(__name__)


def upsert_many(db_path: str, analyses: list[AiAnalysis]) -> int:
    """INSERT OR REPLACE rows. Synchronous — call from the worker thread."""
    if not analyses:
        return 0
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.executemany(
            """INSERT OR REPLACE INTO ai_analysis
               (symbol, as_of_date, interpretation, risk_flags_json, stance, model, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            [
                (
                    _sym_of(a), a.as_of_date, a.interpretation,
                    json.dumps([f.model_dump() for f in a.risk_flags], ensure_ascii=False),
                    a.stance, a.model, a.status,
                )
                for a in analyses
            ],
        )
        conn.commit()
        return len(analyses)
    finally:
        conn.close()


# AiAnalysis has no symbol field (it's the value); the worker pairs it with a
# symbol. Store it transiently via a private attr set by the worker.
def _sym_of(a: AiAnalysis) -> str:
    return getattr(a, "_symbol", "")


async def fetch_analyses(
    session: AsyncSession, symbols: list[str], as_of_date: str,
) -> dict[str, AiAnalysis]:
    """Read non-failed analyses for these symbols at as_of_date."""
    if not symbols or not as_of_date:
        return {}
    res = await session.execute(
        select(AiAnalysisORM).where(
            AiAnalysisORM.symbol.in_(symbols),
            AiAnalysisORM.as_of_date == as_of_date,
            AiAnalysisORM.status != "failed",
        )
    )
    out: dict[str, AiAnalysis] = {}
    for row in res.scalars().all():
        try:
            flags = [RiskFlag(**f) for f in json.loads(row.risk_flags_json or "[]")]
        except Exception:
            flags = []
        out[row.symbol] = AiAnalysis(
            interpretation=row.interpretation, risk_flags=flags, stance=row.stance,
            model=row.model, as_of_date=row.as_of_date, status=row.status,
        )
    return out
```

> Note: `AiAnalysis` carries no `symbol` (it's the served value, keyed by symbol externally). The worker sets a transient `a._symbol` before calling `upsert_many` (Pydantic allows attribute assignment on instances). The test above passes symbol via the `_mk` helper's `interpretation`; adjust `_mk` to set `_symbol`:

Update the test helper `_mk` to set the symbol the store reads:

```python
def _mk(symbol, status="ok"):
    a = AiAnalysis(interpretation=f"note-{symbol}", stance="neutral",
                   model="claude-opus-4-8", as_of_date="2026-06-10", status=status,
                   risk_flags=[RiskFlag(type="解禁", severity="low", reason="r",
                                        source="s", source_date="2026-06-09")])
    object.__setattr__(a, "_symbol", symbol)
    return a
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/analysis/store.py backend/app/analysis/tests/test_store.py
git commit -m "feat(ai-analysis): sqlite upsert (worker) + async serving read"
```

---

## Task 8: Job orchestration (trigger + background worker)

**Files:**
- Create: `backend/app/analysis/service.py`
- Test: `backend/app/analysis/tests/test_service.py`

> Mirrors `inference/service.py`: module-level job dict + lock + daemon thread. The worker derives top-N from `models.service.candidates()` inside the thread (keeps the event loop free), fetches+analyzes each pick with bounded concurrency, then upserts.

- [ ] **Step 1: Write the failing test**

```python
# backend/app/analysis/tests/test_service.py
import time
from app.analysis import service
from app.analysis.schemas import AiAnalysis


def test_disabled_returns_disabled(monkeypatch):
    monkeypatch.setattr(service, "_is_enabled", lambda s: False)
    resp = service.trigger_analysis(reason="manual_ui")
    assert resp.status == "disabled"
    assert service.get_active_job() is None


def test_trigger_runs_worker_and_persists(monkeypatch, tmp_path):
    db = tmp_path / "app.db"
    # Pretend enabled
    monkeypatch.setattr(service, "_is_enabled", lambda s: True)
    monkeypatch.setattr(service, "_db_path", lambda s: str(db))
    monkeypatch.setattr(service, "_model", lambda s: "claude-opus-4-8")
    monkeypatch.setattr(service, "_top_n", lambda s: 2)
    # Fake the picks source
    monkeypatch.setattr(service, "_load_picks", lambda: (
        "2026-06-10",
        [("SH600519", "贵州茅台", {"score_today": 0.9}),
         ("SZ000001", "平安银行", {"score_today": 0.8})],
    ))
    # Fake per-symbol analysis (no network)
    def fake_one(symbol, name, ctx, model, as_of):
        a = AiAnalysis(interpretation=f"n-{symbol}", stance="neutral",
                       model=model, as_of_date=as_of, status="ok")
        return a
    monkeypatch.setattr(service, "_analyze_symbol", fake_one)
    captured = {}
    monkeypatch.setattr(service.store, "upsert_many",
                        lambda path, rows: captured.update({"n": len(rows), "path": path}))

    resp = service.trigger_analysis(reason="manual_ui")
    assert resp.status == "started"
    # wait for the daemon thread
    for _ in range(50):
        job = service.get_job(resp.job_id)
        if job and job.status == "done":
            break
        time.sleep(0.05)
    job = service.get_job(resp.job_id)
    assert job.status == "done"
    assert job.analyzed == 2
    assert captured["n"] == 2
    assert service.get_status().is_running is False


def test_double_trigger_is_already_running(monkeypatch):
    monkeypatch.setattr(service, "_is_enabled", lambda s: True)
    monkeypatch.setattr(service, "_load_picks", lambda: ("2026-06-10", []))
    # Force a slow worker so the second trigger overlaps
    import threading
    gate = threading.Event()
    monkeypatch.setattr(service, "_run_picks", lambda *a, **k: gate.wait(2))
    r1 = service.trigger_analysis(reason="manual_ui")
    r2 = service.trigger_analysis(reason="manual_ui")
    gate.set()
    assert r1.status == "started"
    assert r2.status == "already_running"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.analysis.service'`.

- [ ] **Step 3: Write the service**

```python
# backend/app/analysis/service.py
"""In-memory job tracking + background worker for AI analysis.
Mirrors app/inference/service.py. Single-process backend assumption."""
from __future__ import annotations

import logging
import os
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from app.analysis import store
from app.analysis.llm import analyze_one, make_client
from app.analysis.schemas import AnalysisJob, AnalysisStatus, TriggerResponse
from app.analysis.sources import fetch_news, fetch_notices
from app.core.config import Settings

log = logging.getLogger(__name__)

_MAX_JOBS = 50
_JOBS: "OrderedDict[str, AnalysisJob]" = OrderedDict()
_ACTIVE_JOB_ID: str | None = None
_LOCK = threading.Lock()
_LAST_RUN_AT: str | None = None
_LAST_SUCCESS_AT: str | None = None
_LAST_ERROR: str | None = None

_CONCURRENCY = 4


# --- config resolvers (monkeypatchable in tests) --------------------------
def _is_enabled(s: Settings) -> bool:
    return bool(s.ai_analysis_enabled and (s.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")))

def _db_path(s: Settings) -> str:
    return str(Path(s.app_db_path).expanduser().resolve())

def _model(s: Settings) -> str:
    return s.ai_model

def _top_n(s: Settings) -> int:
    return s.ai_analysis_top_n


def _remember(job_id: str, job: AnalysisJob) -> None:
    _JOBS[job_id] = job
    _JOBS.move_to_end(job_id)
    while len(_JOBS) > _MAX_JOBS:
        _JOBS.popitem(last=False)


def get_active_job() -> AnalysisJob | None:
    if _ACTIVE_JOB_ID and _ACTIVE_JOB_ID in _JOBS:
        return _JOBS[_ACTIVE_JOB_ID]
    return None


def get_status() -> AnalysisStatus:
    return AnalysisStatus(last_run_at=_LAST_RUN_AT, last_success_at=_LAST_SUCCESS_AT,
                          last_error=_LAST_ERROR, is_running=_ACTIVE_JOB_ID is not None)


def get_job(job_id: str) -> AnalysisJob | None:
    return _JOBS.get(job_id)


def _load_picks() -> tuple[str, list[tuple[str, str, dict]]]:
    """Derive (as_of_date, [(symbol, name, context), ...]) from the served candidates.
    Runs inside the worker thread (candidates() is sync + qlib-heavy)."""
    from app.models import service as models_service
    result = models_service.candidates()
    items = result["items"][: _top_n(Settings())]
    as_of = result.get("as_of_date") or result.get("latest_date") or ""
    picks = [
        (it.symbol, it.name, {
            "score_today": it.score_today,
            "pct_change_5d": it.pct_change_5d,
            "board": it.board,
            "is_st": it.is_st,
        })
        for it in items
    ]
    return as_of, picks


def _analyze_symbol(symbol: str, name: str, ctx: dict, model: str, as_of: str):
    """Fetch + LLM for one pick. Returns AiAnalysis (status set) or None on hard error."""
    client = make_client(Settings().anthropic_api_key)
    news = fetch_news(symbol)
    notices = fetch_notices(symbol)
    for attempt in (1, 2):
        try:
            a = analyze_one(client, symbol=symbol, name=name, news=news, notices=notices,
                            context=ctx, model=model, as_of_date=as_of)
            object.__setattr__(a, "_symbol", symbol)
            return a
        except Exception as exc:
            log.warning("analyze_failed symbol=%s attempt=%d: %s", symbol, attempt, exc)
    return None


def _run_picks(job_id: str, db_path: str, model: str) -> int:
    """Worker body — overridable in tests. Returns count analyzed."""
    as_of, picks = _load_picks()
    if not picks:
        return 0
    rows = []
    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as ex:
        futs = [ex.submit(_analyze_symbol, sym, name, ctx, model, as_of)
                for sym, name, ctx in picks]
        for f in futs:
            a = f.result()
            if a is not None:
                rows.append(a)
    store.upsert_many(db_path, rows)
    with _LOCK:
        j = _JOBS.get(job_id)
        if j:
            j.as_of_date = as_of
    return len(rows)


def trigger_analysis(reason: str = "manual_ui") -> TriggerResponse:
    global _ACTIVE_JOB_ID, _LAST_RUN_AT
    s = Settings()
    if not _is_enabled(s):
        return TriggerResponse(status="disabled", job_id=None)

    with _LOCK:
        if _ACTIVE_JOB_ID and _ACTIVE_JOB_ID in _JOBS \
           and _JOBS[_ACTIVE_JOB_ID].status == "running":
            return TriggerResponse(status="already_running", job_id=_ACTIVE_JOB_ID)
        job_id = uuid.uuid4().hex[:12]
        now = datetime.utcnow().isoformat()
        _remember(job_id, AnalysisJob(job_id=job_id, status="running",
                                      started_at=now, reason=reason))
        _ACTIVE_JOB_ID = job_id
        _LAST_RUN_AT = now

    db_path, model = _db_path(s), _model(s)
    threading.Thread(target=_worker, args=(job_id, db_path, model), daemon=True).start()
    return TriggerResponse(status="started", job_id=job_id)


def _worker(job_id: str, db_path: str, model: str) -> None:
    global _ACTIVE_JOB_ID, _LAST_SUCCESS_AT, _LAST_ERROR
    analyzed = None
    err = None
    try:
        analyzed = _run_picks(job_id, db_path, model)
    except Exception as exc:
        log.exception("analysis_worker_error job_id=%s: %s", job_id, exc)
        err = str(exc)[-2000:]
    finally:
        with _LOCK:
            j = _JOBS.get(job_id)
            if j:
                j.status = "failed" if err else "done"
                j.finished_at = datetime.utcnow().isoformat()
                j.analyzed = analyzed
                if err:
                    j.error = err
                    _LAST_ERROR = err
                else:
                    _LAST_SUCCESS_AT = j.finished_at
                    _LAST_ERROR = None
            _ACTIVE_JOB_ID = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_service.py -v`
Expected: PASS. (If `test_double_trigger` flakes, the `_run_picks` monkeypatch + gate is the slow-worker stand-in; it asserts the lock path, not real work.)

- [ ] **Step 5: Commit**

```bash
git add backend/app/analysis/service.py backend/app/analysis/tests/test_service.py
git commit -m "feat(ai-analysis): job tracker + background worker (top-N from candidates)"
```

---

## Task 9: REST endpoints (internal refresh, run-now, status, get)

**Files:**
- Create: `backend/app/analysis/router.py`
- Test: `backend/app/analysis/tests/test_router.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/app/analysis/tests/test_router.py
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.analysis.router import router, internal_router
from app.analysis import service
from app.analysis.schemas import TriggerResponse, AnalysisStatus


def _client():
    app = FastAPI()
    app.include_router(router)
    app.include_router(internal_router)
    return TestClient(app)


def test_run_now_delegates(monkeypatch):
    monkeypatch.setattr(service, "trigger_analysis",
                        lambda reason="manual_ui": TriggerResponse(status="started", job_id="j1"))
    r = _client().post("/api/analysis/run-now")
    assert r.status_code == 200
    assert r.json()["status"] == "started"


def test_status_endpoint(monkeypatch):
    monkeypatch.setattr(service, "get_status",
                        lambda: AnalysisStatus(is_running=True, last_run_at="t"))
    r = _client().get("/api/analysis/status")
    assert r.json()["is_running"] is True


def test_internal_refresh_rejects_non_localhost(monkeypatch):
    monkeypatch.setattr(service, "trigger_analysis",
                        lambda reason="data_refresh": TriggerResponse(status="started", job_id="j2"))
    # TestClient host is "testclient" which IS allowed; simulate a foreign host via header check
    c = _client()
    r = c.post("/api/internal/analysis/refresh")
    assert r.status_code == 200  # testclient allowed
    assert r.json()["status"] == "started"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.analysis.router'`.

- [ ] **Step 3: Write the router**

```python
# backend/app/analysis/router.py
"""AI analysis REST endpoints.

Public:
  GET  /api/analysis/active/peek    -> AnalysisJob | null
  GET  /api/analysis/status         -> AnalysisStatus
  GET  /api/analysis/jobs/{job_id}  -> AnalysisJob
  GET  /api/analysis/{symbol}       -> AiAnalysis | null   (latest stored)
  POST /api/analysis/run-now        -> TriggerResponse

Localhost-only (called by daily_inference subprocess):
  POST /api/internal/analysis/refresh -> TriggerResponse
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis import service
from app.analysis.orm import AiAnalysisORM
from app.analysis.schemas import AiAnalysis, AnalysisJob, AnalysisStatus, TriggerResponse
from app.analysis.store import fetch_analyses
from app.core.db import get_session

router = APIRouter(prefix="/api/analysis", tags=["analysis"])
internal_router = APIRouter(prefix="/api/internal", tags=["internal"])


@router.get("/active/peek", response_model=AnalysisJob | None)
def active_peek():
    return service.get_active_job()


@router.get("/status", response_model=AnalysisStatus)
def status():
    return service.get_status()


@router.get("/jobs/{job_id}", response_model=AnalysisJob)
def get_job(job_id: str):
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(404, detail="job not found")
    return job


@router.post("/run-now", response_model=TriggerResponse)
def run_now():
    return service.trigger_analysis(reason="manual_ui")


@router.get("/{symbol}", response_model=AiAnalysis | None)
async def get_for_symbol(symbol: str, session: AsyncSession = Depends(get_session)):
    # newest stored row for this symbol (any date)
    res = await session.execute(
        select(AiAnalysisORM).where(AiAnalysisORM.symbol == symbol,
                                    AiAnalysisORM.status != "failed")
        .order_by(AiAnalysisORM.as_of_date.desc()).limit(1)
    )
    row = res.scalars().first()
    if row is None:
        return None
    got = await fetch_analyses(session, [symbol], row.as_of_date)
    return got.get(symbol)


@internal_router.post("/analysis/refresh", response_model=TriggerResponse)
def internal_refresh(request: Request):
    client_host = request.client.host if request.client else None
    if client_host not in ("127.0.0.1", "localhost", "::1", "testclient"):
        raise HTTPException(403, detail="localhost only")
    return service.trigger_analysis(reason="data_refresh")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_router.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/analysis/router.py backend/app/analysis/tests/test_router.py
git commit -m "feat(ai-analysis): REST endpoints (run-now, status, internal refresh, get)"
```

---

## Task 10: Register routers in the app

**Files:**
- Modify: `backend/app/main.py`
- Test: `backend/app/analysis/tests/test_app_wiring.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/app/analysis/tests/test_app_wiring.py
def test_analysis_routes_registered():
    from app.main import create_app
    app = create_app()
    paths = {r.path for r in app.routes}
    assert "/api/analysis/status" in paths
    assert "/api/internal/analysis/refresh" in paths
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_app_wiring.py -v`
Expected: FAIL — `/api/analysis/status` not in paths.

- [ ] **Step 3: Register the routers**

In `backend/app/main.py`, add the import near the other routers (after the inference import line):

```python
from app.analysis.router import router as analysis_router, internal_router as analysis_internal_router
```

In `create_app()`, after `app.include_router(inference_internal_router)`:

```python
    app.include_router(analysis_router)
    app.include_router(analysis_internal_router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_app_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/main.py backend/app/analysis/tests/test_app_wiring.py
git commit -m "feat(ai-analysis): register analysis routers"
```

---

## Task 11: Serving join — attach ai_analysis to candidates

**Files:**
- Modify: `backend/app/models/schemas.py` (add field)
- Modify: `backend/app/models/router.py` (attach in candidates route)
- Test: `backend/app/analysis/tests/test_candidates_join.py`

> Attach at the **router** (not the lru-cached `_candidates_cached`) so analysis stays fresh and the qlib cache isn't polluted. Use `model_copy` so cached `ScreenItem` objects aren't mutated.

- [ ] **Step 1: Write the failing test**

```python
# backend/app/analysis/tests/test_candidates_join.py
import pytest
from app.models.schemas import ScreenItem
from app.analysis.schemas import AiAnalysis


def test_screenitem_has_ai_analysis_field():
    it = ScreenItem(rank=1, symbol="SH600519", score_today=0.9, score_avg=0.9,
                    rank_avg=1.0, days_in_top=1)
    assert it.ai_analysis is None
    it2 = it.model_copy(update={"ai_analysis": AiAnalysis(interpretation="x", stance="neutral")})
    assert it2.ai_analysis.interpretation == "x"
    assert it.ai_analysis is None  # original untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_candidates_join.py -v`
Expected: FAIL — `ScreenItem` has no `ai_analysis`.

- [ ] **Step 3: Add the field + attach in the route**

In `backend/app/models/schemas.py`, add the import at top and the field on `ScreenItem` (after `horizons`):

```python
from app.analysis.schemas import AiAnalysis
```
```python
    # AI 分析层 (解读 + 风险旗标) — attached at serving time, None when not yet generated
    ai_analysis: AiAnalysis | None = None
```

In `backend/app/models/router.py`, change the candidates endpoint to async + attach. Replace the imports line and the `candidates_endpoint`:

```python
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.db import get_session
from app.analysis.store import fetch_analyses
```
```python
@router.get("/candidates", response_model=CandidatesResponse)
async def candidates_endpoint(
    top: int = Query(default=300, ge=1, le=500),
    days: int = Query(default=5, ge=1, le=60),
    min_top: int = Query(default=0, ge=0),
    experiment: str | None = Query(default=None),
    view: str = Query(default="ensemble", pattern="^(ensemble|lightgbm|alstm|tra)$"),
    models: str | None = Query(default=None, description="(unchanged — see below)"),
    session: AsyncSession = Depends(get_session),
):
    """Return the full candidate pool (cached per recorder + view + models + base params),
    with AI analysis (解读 + 风险旗标) LEFT-JOINed onto each item by (symbol, latest_date)."""
    result = service.candidates(
        top=top, days=days, min_top=min_top, experiment=experiment,
        view=view, models=models,
    )
    items = result["items"]
    as_of = result.get("as_of_date") or result.get("latest_date") or ""
    analyses = await fetch_analyses(session, [it.symbol for it in items], as_of)
    if analyses:
        result["items"] = [
            it.model_copy(update={"ai_analysis": analyses.get(it.symbol)}) for it in items
        ]
    return result
```

> Keep the original `models` Query description text (copy it verbatim from the current file — it's a multi-line help string). Only the signature shape (`async`, `Depends(get_session)`) and the body change.

- [ ] **Step 4: Run tests (unit + the integration join)**

Add an integration test in the same file that monkeypatches `fetch_analyses` and `service.candidates`:

```python
@pytest.mark.asyncio
async def test_candidates_route_attaches_analysis(monkeypatch):
    from fastapi import FastAPI
    from httpx import AsyncClient, ASGITransport
    from app.models import router as models_router

    monkeypatch.setattr(models_router.service, "candidates", lambda **kw: {
        "experiment": "e", "recorder_id": "r", "latest_date": "2026-06-10",
        "window_days": 5, "universe_size": 1, "available_models": [],
        "items": [ScreenItem(rank=1, symbol="SH600519", score_today=1.0,
                             score_avg=1.0, rank_avg=1.0, days_in_top=1)],
        "as_of_date": "2026-06-10",
    })

    async def fake_fetch(session, symbols, as_of):
        return {"SH600519": AiAnalysis(interpretation="超跌反弹", stance="favorable")}
    monkeypatch.setattr(models_router, "fetch_analyses", fake_fetch)

    async def _noop_session():
        yield None
    app = FastAPI()
    app.dependency_overrides = {}
    app.include_router(models_router.router, prefix="/api/models")
    from app.core.db import get_session
    app.dependency_overrides[get_session] = _noop_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.get("/api/models/candidates")
    body = r.json()
    assert body["items"][0]["ai_analysis"]["interpretation"] == "超跌反弹"
```

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/analysis/tests/test_candidates_join.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Run the existing models tests (regression)**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/models/tests/ -v`
Expected: PASS — `ai_analysis` defaults to `None`, existing shape unchanged. Fix any test that constructs `CandidatesResponse`/`ScreenItem` positionally (none expected).

- [ ] **Step 6: Commit**

```bash
git add backend/app/models/schemas.py backend/app/models/router.py backend/app/analysis/tests/test_candidates_join.py
git commit -m "feat(ai-analysis): attach ai_analysis to candidates at serving time"
```

---

## Task 12: Trigger analysis from daily_inference

**Files:**
- Modify: `production/daily_inference.py`
- Test: `production/tests/test_daily_inference_analysis_trigger.py` (create; if `production/tests/` lacks `__init__.py`, add it)

- [ ] **Step 1: Write the failing test**

```python
# production/tests/test_daily_inference_analysis_trigger.py
import production.daily_inference as di


def test_post_analysis_refresh_posts_to_internal_url(monkeypatch):
    calls = {}

    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=0):
        calls["url"] = req.full_url
        calls["method"] = req.get_method()
        return _Resp()

    monkeypatch.setattr(di.urllib.request, "urlopen", fake_urlopen)
    di._post_analysis_refresh()
    assert calls["url"] == "http://127.0.0.1:8000/api/internal/analysis/refresh"
    assert calls["method"] == "POST"


def test_post_analysis_refresh_failsoft(monkeypatch):
    def boom(req, timeout=0): raise OSError("backend down")
    monkeypatch.setattr(di.urllib.request, "urlopen", boom)
    di._post_analysis_refresh()  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/Projects/qlib && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_daily_inference_analysis_trigger.py -v`
Expected: FAIL — `_post_analysis_refresh` not defined (and possibly `di.urllib` not imported at module scope).

- [ ] **Step 3: Add the trigger**

In `production/daily_inference.py`, near the existing `INVALIDATE_URL` constant (line ~49), add:

```python
ANALYSIS_REFRESH_URL = "http://127.0.0.1:8000/api/internal/analysis/refresh"
```

Ensure `import urllib.request` is at module scope (the existing `_post_invalidate_cache` imports it inside the function — add a top-level `import urllib.request` so the test can monkeypatch `di.urllib.request`). Then add, right after `_post_invalidate_cache`:

```python
def _post_analysis_refresh():
    """Best-effort: ask the backend to (re)generate AI analysis for the top-N picks.
    No-op server-side when ai_analysis_enabled is false / no key."""
    try:
        req = urllib.request.Request(ANALYSIS_REFRESH_URL, method="POST")
        with urllib.request.urlopen(req, timeout=3) as r:
            log.info("analysis_refresh status=%d", r.status)
    except Exception as exc:
        log.warning("analysis_refresh_failed: %s — backend may be down", exc)
```

In `run()`, right after the existing `_post_invalidate_cache()` (line ~374):

```python
    _post_invalidate_cache()
    _post_analysis_refresh()
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/Projects/qlib && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_daily_inference_analysis_trigger.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add production/daily_inference.py production/tests/test_daily_inference_analysis_trigger.py
git commit -m "feat(ai-analysis): trigger analysis refresh after daily_inference appends"
```

---

## Task 13: Backend full-suite regression + manual smoke

**Files:** none (verification)

- [ ] **Step 1: Run the whole backend suite**

Run: `cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest -q`
Expected: all pass (analysis module + no regressions in charts/models/inference/portfolio/scheduling).

- [ ] **Step 2: Live smoke (real key, top-2)**

Set env and start the backend from the MAIN repo:
```bash
$env:QLIB_COMPANION_ANTHROPIC_API_KEY="sk-ant-..."
$env:QLIB_COMPANION_AI_ANALYSIS_ENABLED="true"
$env:QLIB_COMPANION_AI_ANALYSIS_TOP_N="2"
cd /e/Projects/qlib/backend; F:/Tools/Anaconda/envs/qlib/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
In another shell:
```bash
curl -s -X POST http://127.0.0.1:8000/api/analysis/run-now
# poll
curl -s http://127.0.0.1:8000/api/analysis/status
# after done, confirm attach:
curl -s "http://localhost:8000/api/models/candidates?top=10" | F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -c "import sys,json;d=json.load(sys.stdin);print([{'s':i['symbol'],'ai':bool(i.get('ai_analysis'))} for i in d['items'][:5]])"
```
Expected: `run-now` → `{"status":"started",...}`; status → `done` with no error; candidates top items show `ai: True` with interpretation + (possibly) flags. If `status:"disabled"`, the env wasn't picked up — check the `QLIB_COMPANION_` prefix.

- [ ] **Step 3: Commit (if any fixes were needed)**

```bash
git add -A && git commit -m "test(ai-analysis): backend suite green + live smoke fixes"
```

---

## Task 14: Frontend — types, job badge, risk-flag badge + note panel

**Files:**
- Regenerate: `frontend/src/api/types.gen.ts` (via `npm run gen:api`)
- Modify: `frontend/src/api/client.ts` (add `analysis` methods)
- Modify: `frontend/src/jobs/useActiveJobs.ts` (add `'analysis'` kind)
- Create: `frontend/src/pages/picks/RiskFlagBadge.tsx`
- Create: `frontend/src/pages/picks/AiNotePanel.tsx`
- Modify: `frontend/src/pages/Picks.tsx` (render badge + expandable note; refetch on analysis-done)

> Frontend follows existing patterns; verify visually rather than via unit tests (matches the repo's frontend convention). Match Tailwind classes used by `HorizonMiniBar.tsx` for visual consistency. A-share color convention: red = good/up, green = caution/down (already used by `PredictionChart`).

- [ ] **Step 1: Regenerate API types from the running backend**

With the backend running (Task 13 Step 2), run: `cd /e/Projects/qlib/frontend && npm run gen:api`
Expected: `types.gen.ts` now contains `AiAnalysis`, `RiskFlag`, `AnalysisJob`, `AnalysisStatus`, and `ScreenItem.ai_analysis`.

- [ ] **Step 2: Add API client methods**

In `frontend/src/api/client.ts`, add an `analysis` group mirroring `inference` (find the `inference` block and copy its shape):

```ts
  analysis: {
    active: () => get<AnalysisJob | null>('/api/analysis/active/peek'),
    status: () => get<AnalysisStatus>('/api/analysis/status'),
    runNow: () => post<TriggerResponse>('/api/analysis/run-now'),
    forSymbol: (symbol: string) => get<AiAnalysis | null>(`/api/analysis/${symbol}`),
  },
```
Import the generated types (`AnalysisJob`, `AnalysisStatus`, `AiAnalysis`, `TriggerResponse`) alongside the existing imports. Use the exact `get`/`post` helpers already in the file.

- [ ] **Step 3: Add the 'analysis' job kind**

In `frontend/src/jobs/useActiveJobs.ts`:
- Extend the union: `export type ActiveJobKind = 'refresh' | 'retrain' | 'evaluation' | 'inference' | 'analysis';`
- Add a poll: `const analysis = useJobPolling('analysis', () => api.analysis.active(), interval);`
- Add `analysis` to the `running` check in the `useEffect` deps and condition: `(analysis?.status === 'running')`.
- Append an `out.push({...})` block mirroring the `inference` one, with `label` `'AI 解读'` / `'✓ 解读完成'` / `'✗ 解读失败'`, `detail` = `analysis.analyzed != null ? `${analysis.analyzed} 只` : undefined`, `href: '/picks'`.

- [ ] **Step 4: Create RiskFlagBadge**

```tsx
// frontend/src/pages/picks/RiskFlagBadge.tsx
import type { AiAnalysis } from '@/api/types.gen';

const SEVERITY_CLASS: Record<string, string> = {
  high: 'bg-red-600 text-white',
  medium: 'bg-amber-500 text-white',
  low: 'bg-zinc-400 text-white',
};

export function RiskFlagBadge({ analysis }: { analysis?: AiAnalysis | null }) {
  if (!analysis || analysis.risk_flags.length === 0) return null;
  const worst = analysis.risk_flags.reduce(
    (acc, f) => (f.severity === 'high' ? 'high' : acc === 'high' ? 'high' : f.severity),
    'low' as string,
  );
  return (
    <span
      className={`inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 text-xs font-medium ${SEVERITY_CLASS[worst] ?? SEVERITY_CLASS.low}`}
      title={analysis.risk_flags.map((f) => `${f.type}: ${f.reason}`).join('\n')}
    >
      🚩 {analysis.risk_flags.length}
    </span>
  );
}
```

- [ ] **Step 5: Create AiNotePanel**

```tsx
// frontend/src/pages/picks/AiNotePanel.tsx
import type { AiAnalysis } from '@/api/types.gen';

const STANCE_CLASS: Record<string, string> = {
  favorable: 'text-red-600',     // A-share: red = positive
  caution: 'text-green-600',     // green = caution/down
  neutral: 'text-zinc-500',
};

export function AiNotePanel({ analysis }: { analysis?: AiAnalysis | null }) {
  if (!analysis) return <p className="text-xs text-zinc-400">暂无当日 AI 解读</p>;
  return (
    <div className="space-y-1 text-sm">
      <p className={STANCE_CLASS[analysis.stance] ?? STANCE_CLASS.neutral}>
        {analysis.interpretation}
      </p>
      {analysis.risk_flags.length > 0 && (
        <ul className="space-y-0.5">
          {analysis.risk_flags.map((f, i) => (
            <li key={i} className="text-xs">
              <span className="font-medium">[{f.type}]</span> {f.reason}
              <span className="text-zinc-400"> — {f.source}（{f.source_date}）</span>
            </li>
          ))}
        </ul>
      )}
      <p className="text-[10px] text-zinc-400">
        数据截至 {analysis.as_of_date} · {analysis.model}
        {analysis.status === 'partial' ? ' · 仅基于上下文' : ''}
      </p>
    </div>
  );
}
```

- [ ] **Step 6: Render in Picks + refetch on analysis-done**

In `frontend/src/pages/Picks.tsx`:
- Import `RiskFlagBadge`, `AiNotePanel`, and the candidates query key + `useQueryClient`.
- In the table row, render `<RiskFlagBadge analysis={item.ai_analysis} />` next to the symbol/name cell.
- Make the row expandable (reuse the existing expand mechanism if one exists, e.g. `RecentlyViewed`/row-detail; otherwise add a click-to-expand `useState<string|null>` for the open symbol) and render `<AiNotePanel analysis={item.ai_analysis} />` in the expanded area.
- Watch the analysis job and refetch candidates when it finishes: poll `api.analysis.active()` (or reuse `useActiveJobs`), and on transition to `done`, call `queryClient.invalidateQueries({ queryKey: [<candidates key>] })` so the fresh `ai_analysis` shows. Use the candidates query key already defined in `frontend/src/models/hooks.ts`.

- [ ] **Step 7: Typecheck**

Run: `cd /e/Projects/qlib/frontend && npm run typecheck`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/api/ frontend/src/jobs/useActiveJobs.ts frontend/src/pages/picks/RiskFlagBadge.tsx frontend/src/pages/picks/AiNotePanel.tsx frontend/src/pages/Picks.tsx
git commit -m "feat(ai-analysis): Picks risk-flag badge + note panel + analysis job badge"
```

---

## Task 15: Restart front + back; visual verification

**Files:** none (standing instruction — UI/server changed).

- [ ] **Step 1: Restart backend (MAIN repo, with the AI env vars set)**

```bash
$env:QLIB_COMPANION_ANTHROPIC_API_KEY="sk-ant-..."; $env:QLIB_COMPANION_AI_ANALYSIS_ENABLED="true"
cd /e/Projects/qlib/backend; F:/Tools/Anaconda/envs/qlib/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- [ ] **Step 2: Restart frontend**

```bash
cd /e/Projects/qlib/frontend; npm run dev   # http://localhost:5173
```

- [ ] **Step 3: Verify in the browser**

- Trigger analysis (POST `/api/analysis/run-now` or click 刷新数据 to run the full chain) → header shows the "AI 解读" job badge.
- On completion, Picks table shows 🚩 badges on flagged picks; expanding a row shows the one-line interpretation (stance-colored) + flag list with sources + the "数据截至 … · model" footnote.
- Toggle `QLIB_COMPANION_AI_ANALYSIS_ENABLED=false` and restart → picks render normally with no badges (fail-soft).

- [ ] **Step 4: Final commit / wrap-up**

```bash
git add -A && git commit -m "chore(ai-analysis): v1 complete — verified end to end"
```

---

## Self-review notes (coverage vs spec)

- ✅ 解读 + 风险旗标 (Task 2/5/6); ✅ 不改 qlib 排名 (serving-join only, Task 11); ✅ 新闻+公告+上下文输入 (Task 4/5); ✅ 方案 A 批量挂 daily_inference (Task 12); ✅ backend owns LLM/DB/serving (Task 6/7/8/11); ✅ Step-0 spike gate (Task 0); ✅ structured output + anti-hallucination (Task 5/6); ✅ fail-soft: disabled/no-key, akshare error, LLM retry→failed, partial status (Task 4/6/8); ✅ idempotent upsert (Task 7); ✅ config knobs incl. top_n cost-cap (Task 1); ✅ TDD units + integration; ✅ v2 deferrals (基本面/资金面/on-demand/ChartPage/multi-agent) not in scope.
- **Carry-overs to confirm during build:** exact akshare column names + announcement interface (Task 0 → Task 4); Alembic head id (Task 3); the existing Picks row-expand mechanism (Task 14); the candidates query key name in `models/hooks.ts` (Task 14).
