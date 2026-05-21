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


@pytest.mark.asyncio
async def test_get_session_raises_after_dispose(tmp_path, monkeypatch):
    monkeypatch.setenv("QLIB_COMPANION_APP_DB_PATH", str(tmp_path / "x.db"))
    from app.core.db import init_db_singletons, dispose_db_singletons, get_session
    from app.core.config import Settings

    init_db_singletons(Settings())
    # sanity: a session can be acquired
    gen = get_session()
    sess = await gen.__anext__()
    await sess.close()
    await gen.aclose()

    # now dispose and verify get_session raises
    await dispose_db_singletons()
    with pytest.raises(RuntimeError, match="not initialized"):
        async for _ in get_session():
            pass
