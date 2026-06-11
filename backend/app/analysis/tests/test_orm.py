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
