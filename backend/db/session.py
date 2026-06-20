from __future__ import annotations

import os
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.db.models import Base

# Single source of truth for the SQLite location (PRD Section 9). The writer here and
# the read-only small-to-big expansion in graph/nodes/expand.py both derive from this var.
SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH", "./backend/data/app.db")
DATABASE_URL = f"sqlite+aiosqlite:///{SQLITE_DB_PATH}"

engine = create_async_engine(DATABASE_URL, echo=False)

async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, expire_on_commit=False
)


async def init_db() -> None:
    # Ensure the SQLite parent directory exists (backend/data is gitignored and
    # absent on a fresh clone, so create_all would fail to open the file).
    parent = os.path.dirname(SQLITE_DB_PATH)
    if parent and ":memory:" not in SQLITE_DB_PATH:
        os.makedirs(parent, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    await engine.dispose()


async def get_session() -> AsyncIterator[AsyncSession]:
    async with async_session() as session:
        yield session
