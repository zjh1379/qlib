import pytest
from sqlalchemy import update

from app.core import db as db_mod
from app.core.config import Settings
from app.core.db import init_db_singletons


@pytest.fixture(autouse=True)
async def reset_retrain_schedule_row():
    """Reset retrain_schedule row 1 to defaults before each test.

    Tests in this module mutate the persistent SQLite app.db. Without this
    fixture the suite is non-idempotent — running it twice fails on the second
    run because row 1 still carries the values written by a prior test.
    """
    if db_mod._session_maker is None:
        init_db_singletons(Settings())

    # Late import — the orm module imports from app.core.db, which we just
    # ensured is initialized.
    from app.scheduling.orm import RetrainScheduleORM

    if db_mod._session_maker is None:
        # init_db_singletons failed for some reason; bail with a clear test failure
        pytest.fail("could not initialize DB singletons for retrain_schedule reset fixture")

    async with db_mod._session_maker() as session:
        await session.execute(
            update(RetrainScheduleORM)
            .where(RetrainScheduleORM.id == 1)
            .values(day_of_week=6, hour=22, minute=0, enabled=True,
                    last_run_at=None, next_run_at=None)
        )
        await session.commit()

    yield
