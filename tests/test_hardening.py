"""
Hardening tests for public hosting: arxiv_id validation, env-driven CORS,
and slowapi rate limiting on the expensive POST /papers endpoint.

External deps (ingest_paper) are stubbed; everything runs against an in-memory
SQLite database, mirroring tests/test_api.py.
"""

from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.api.app import create_app
from backend.db.models import Base
from backend.db.session import get_session

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture()
async def engine():
    eng = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture()
async def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture()
async def client(session_factory):
    async def override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_session] = override_get_session

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# arxiv_id validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["not-an-id", "../etc", "12.3", "", "2301.1"])
async def test_post_papers_invalid_arxiv_id_returns_422(client, bad_id):
    with patch("backend.api.routers.papers.ingest_paper"):
        resp = await client.post("/papers", json={"arxiv_id": bad_id})
    assert resp.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "good_id",
    ["2301.00001", "2301.00001v2", "1706.03762", "hep-th/9901001", "math.AG/0309001v2"],
)
async def test_post_papers_valid_arxiv_id_passes_validation(client, good_id):
    with patch("backend.api.routers.papers.ingest_paper"):
        resp = await client.post("/papers", json={"arxiv_id": good_id})
    # 202 Accepted means validation passed and ingestion was queued.
    assert resp.status_code == 202
    assert resp.json()["arxiv_id"] == good_id


# ---------------------------------------------------------------------------
# CORS — env-driven origins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cors_uses_frontend_origin_env_var(monkeypatch):
    monkeypatch.setenv(
        "FRONTEND_ORIGIN", "https://ferret.fly.dev, https://example.com"
    )
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.options(
            "/papers",
            headers={
                "Origin": "https://ferret.fly.dev",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == "https://ferret.fly.dev"


@pytest.mark.asyncio
async def test_cors_falls_back_to_localhost_when_env_unset(monkeypatch):
    monkeypatch.delenv("FRONTEND_ORIGIN", raising=False)
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.options(
            "/papers",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_limiter_attached_to_app_state():
    from backend.api.limiter import limiter

    app = create_app()
    assert app.state.limiter is limiter
