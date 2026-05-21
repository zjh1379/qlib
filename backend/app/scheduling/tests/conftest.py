import pytest
from sqlalchemy import update

from app.core.db import _session_maker, init_db_singletons
from app.core.config import Settings


@pytest.fixture(autouse=True)
async def reset_retrain_schedule_row():
    """Reset retrain_schedule row 1 to defaults before each test.

    Tests in this module mutate the persistent SQLite app.db. Without this
    fixture the suite is non-idempotent — running it twice fails on the second
    run because row 1 still carries the values written by a prior test.
    """
    # The DB singletons are initialized inside the FastAPI lifespan by each
    # test's client fixture. For tests that don't use that fixture (or for
    # the reset-before phase) we may not have a session maker yet — fall back
    # to a one-shot init from Settings.
    if _session_maker is None:
        init_db_singletons(Settings())

    from app.scheduling.orm import RetrainScheduleORM

    # The session maker was just (re)initialized — re-import to get the live
    # reference (the module-level `_session_maker` is shadowed at function scope).
    from app.core import db as db_mod

    if db_mod._session_maker is None:
        # Bail silently if init failed; the test will likely fail on its own
        # with a clear error.
        yield
        return

    async with db_mod._session_maker() as session:
        await session.execute(
            update(RetrainScheduleORM)
            .where(RetrainScheduleORM.id == 1)
            .values(day_of_week=6, hour=22, minute=0, enabled=True,
                    last_run_at=None, next_run_at=None)
        )
        await session.commit()

    yield

    # No post-test cleanup needed — next test's pre-fixture will reset again.
