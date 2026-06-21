"""Client-ID-based session isolation (X-Client-ID header).

Anonymous per-browser scoping: sessions created with one client id must not be
visible to another. Reuses the in-memory DB + ASGI client pattern from test_api.py.
"""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.api.app import create_app
from backend.db.models import Base, Session as SessionModel
from backend.db.session import get_session

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

CLIENT_A = "client-aaaaaaaa"
CLIENT_B = "client-bbbbbbbb"


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
# GET /sessions — scoping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_sessions_returns_only_own_client_sessions(client):
    resp = await client.post(
        "/sessions", json={"mode": "ask"}, headers={"X-Client-ID": CLIENT_A}
    )
    assert resp.status_code == 201
    sid = resp.json()["session_id"]

    # Client A sees it.
    resp_a = await client.get("/sessions", headers={"X-Client-ID": CLIENT_A})
    assert resp_a.status_code == 200
    ids_a = [s["session_id"] for s in resp_a.json()]
    assert sid in ids_a

    # Client B does not.
    resp_b = await client.get("/sessions", headers={"X-Client-ID": CLIENT_B})
    assert resp_b.status_code == 200
    ids_b = [s["session_id"] for s in resp_b.json()]
    assert sid not in ids_b


@pytest.mark.asyncio
async def test_get_sessions_without_header_returns_empty(client):
    # Create a session owned by client A so the global table is non-empty.
    await client.post(
        "/sessions", json={"mode": "ask"}, headers={"X-Client-ID": CLIENT_A}
    )

    resp = await client.get("/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /sessions/{id}/messages — ownership check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_messages_wrong_client_returns_404(client):
    resp = await client.post(
        "/sessions", json={"mode": "ask"}, headers={"X-Client-ID": CLIENT_A}
    )
    sid = resp.json()["session_id"]

    resp_b = await client.get(
        f"/sessions/{sid}/messages", headers={"X-Client-ID": CLIENT_B}
    )
    assert resp_b.status_code == 404


@pytest.mark.asyncio
async def test_get_messages_correct_client_succeeds(client):
    resp = await client.post(
        "/sessions", json={"mode": "ask"}, headers={"X-Client-ID": CLIENT_A}
    )
    sid = resp.json()["session_id"]

    resp_a = await client.get(
        f"/sessions/{sid}/messages", headers={"X-Client-ID": CLIENT_A}
    )
    assert resp_a.status_code == 200
    assert resp_a.json() == []


@pytest.mark.asyncio
async def test_get_messages_legacy_null_client_session_allows_access(
    client, session_factory
):
    # Simulate a legacy row written before client_id existed (client_id is None).
    async with session_factory() as db:
        legacy = SessionModel(mode="ask", client_id=None)
        db.add(legacy)
        await db.commit()
        await db.refresh(legacy)
        sid = legacy.session_id

    # Any client (or none) may read a null-client legacy session.
    resp = await client.get(
        f"/sessions/{sid}/messages", headers={"X-Client-ID": CLIENT_B}
    )
    assert resp.status_code == 200
    assert resp.json() == []
