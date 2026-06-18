from __future__ import annotations

import os
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.db.models import Base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./ferret.db")

engine = create_async_engine(DATABASE_URL, echo=False)

async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, expire_on_commit=False
)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    await engine.dispose()


async def get_session() -> AsyncIterator[AsyncSession]:
    async with async_session() as session:
        yield session


def session_context():
    """Return an async context manager yielding a fresh AsyncSession.

    Use outside the FastAPI dependency lifecycle (background tasks, SSE generators).
    """
    return async_session()
