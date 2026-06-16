import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.db import Base
from app.analysis import store
from app.analysis.orm import AiAnalysisORM  # noqa: F401 (register table on Base.metadata)
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

    # sync write (as the worker thread does): list of (symbol, AiAnalysis) tuples
    store.upsert_many(str(db), [("SH600519", _mk("SH600519")), ("SZ000001", _mk("SZ000001"))])
    store.upsert_many(str(db), [("SH600519", _mk("SH600519"))])  # idempotent overwrite

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
    store.upsert_many(str(db), [("SH600519", _mk("SH600519", status="failed"))])
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        got = await store.fetch_analyses(s, ["SH600519"], "2026-06-10")
    assert got == {}
    await engine.dispose()


@pytest.mark.asyncio
async def test_fetch_empty_inputs(tmp_path):
    db = tmp_path / "app.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        assert await store.fetch_analyses(s, [], "2026-06-10") == {}
        assert await store.fetch_analyses(s, ["SH600519"], "") == {}
    await engine.dispose()


@pytest.mark.asyncio
async def test_existing_ok_symbols_filters_by_status_and_date(tmp_path):
    db = tmp_path / "app.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    store.upsert_many(str(db), [
        ("SH600519", _mk("SH600519", status="ok")),
        ("SZ000001", _mk("SZ000001", status="partial")),  # partial -> retryable, not skipped
    ])
    other = _mk("SH600036", status="ok")
    other.as_of_date = "2026-06-09"                        # different date
    store.upsert_many(str(db), [("SH600036", other)])

    got = store.existing_ok_symbols(str(db), "2026-06-10")
    assert got == {"SH600519"}
    await engine.dispose()


def test_existing_ok_symbols_missing_table_returns_empty(tmp_path):
    # No table created — must degrade to "nothing done yet", not raise.
    db = tmp_path / "app.db"
    assert store.existing_ok_symbols(str(db), "2026-06-10") == set()


@pytest.mark.asyncio
async def test_fetch_analyses_missing_table_returns_empty(tmp_path):
    # Table NOT created (no create_all) — simulates pre-migration environment.
    db = tmp_path / "app.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        got = await store.fetch_analyses(s, ["SH600519"], "2026-06-10")
    assert got == {}   # degrades gracefully instead of raising
    await engine.dispose()
