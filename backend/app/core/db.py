from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import Settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models. Tables defined in module models.py files."""


def create_engine_and_session(
    settings: Settings | None = None,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    settings = settings or Settings()
    engine = create_async_engine(settings.db_url, echo=False, future=True)
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return engine, session_maker


# Singletons populated at startup in main.py lifespan
_engine: AsyncEngine | None = None
_session_maker: async_sessionmaker[AsyncSession] | None = None


def init_db_singletons(settings: Settings) -> None:
    global _engine, _session_maker
    _engine, _session_maker = create_engine_and_session(settings)


async def dispose_db_singletons() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


async def get_session() -> AsyncIterator[AsyncSession]:
    if _session_maker is None:
        raise RuntimeError("DB not initialized; call init_db_singletons() at startup")
    async with _session_maker() as session:
        yield session
