"""Async SQLAlchemy engine and session dependency for FastAPI."""

import os
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.db.models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./ferret.db")

_engine = create_async_engine(DATABASE_URL, echo=False)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def init_db() -> None:
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    await _engine.dispose()


async def get_session() -> AsyncIterator[AsyncSession]:
    async with _session_factory() as session:
        yield session
