import pytest
from sqlalchemy import text
from app.core.db import create_engine_and_session, get_session


@pytest.mark.asyncio
async def test_session_factory_yields_working_session(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("QLIB_COMPANION_APP_DB_PATH", str(db_path))

    engine, session_maker = create_engine_and_session()
    async with session_maker() as session:
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1
    await engine.dispose()
