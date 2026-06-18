"""
Module 4 unit tests — FastAPI + SSE.

Modules 1 (ingest_paper) and 3 (astream_chat) are stubbed via pytest-mock.
All tests run against an in-memory SQLite database (no real file I/O).
"""

import json
import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.api.app import create_app
from backend.db.models import Base
from backend.db.session import get_session

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    """AsyncClient wired to a fresh in-memory DB for each test."""

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
# Helper: collect SSE events from a streaming response
# ---------------------------------------------------------------------------

async def collect_sse(response) -> list[dict]:
    """Parse raw SSE bytes into a list of {event, data} dicts."""
    events = []
    current: dict = {}
    async for line in response.aiter_lines():
        line = line.strip()
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current["data"] = json.loads(line[len("data:"):].strip())
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


# ---------------------------------------------------------------------------
# POST /papers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_papers_queues_background_ingestion_and_returns_status(client):
    with patch("backend.api.routers.papers.ingest_paper") as mock_ingest:
        mock_ingest.return_value = None
        resp = await client.post("/papers", json={"arxiv_id": "2301.00001"})

    assert resp.status_code == 202
    body = resp.json()
    assert body["arxiv_id"] == "2301.00001"
    assert body["ingestion_status"] == "pending"


@pytest.mark.asyncio
async def test_post_papers_idempotent_for_already_ingested_paper(client):
    with patch("backend.api.routers.papers.ingest_paper"):
        await client.post("/papers", json={"arxiv_id": "2301.00002"})
        resp = await client.post("/papers", json={"arxiv_id": "2301.00002"})

    assert resp.status_code == 202
    assert resp.json()["arxiv_id"] == "2301.00002"


@pytest.mark.asyncio
async def test_post_papers_invalid_arxiv_id_returns_4xx(client):
    resp = await client.post("/papers", json={"arxiv_id": ""})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /papers/{arxiv_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_papers_status_returns_current_ingestion_status(client):
    with patch("backend.api.routers.papers.ingest_paper"):
        await client.post("/papers", json={"arxiv_id": "2301.00003"})

    resp = await client.get("/papers/2301.00003")
    assert resp.status_code == 200
    body = resp.json()
    assert body["arxiv_id"] == "2301.00003"
    assert "ingestion_status" in body


@pytest.mark.asyncio
async def test_get_papers_unknown_arxiv_id_returns_404(client):
    resp = await client.get("/papers/9999.99999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_sessions_deep_dive_requires_paper_id(client):
    resp = await client.post("/sessions", json={"mode": "deep_dive"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_post_sessions_ask_does_not_require_paper_id(client):
    resp = await client.post("/sessions", json={"mode": "ask"})
    assert resp.status_code == 201
    assert "session_id" in resp.json()


@pytest.mark.asyncio
async def test_post_sessions_writes_session_row_to_sqlite(client, session_factory):
    from sqlalchemy import select

    from backend.db.models import Session as SessionModel

    resp = await client.post("/sessions", json={"mode": "ask"})
    assert resp.status_code == 201
    sid = resp.json()["session_id"]

    async with session_factory() as db:
        row = await db.get(SessionModel, sid)

    assert row is not None
    assert row.mode == "ask"


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/messages  — SSE stream
# ---------------------------------------------------------------------------

MOCK_STREAM_EVENTS = [
    {"type": "token", "content": "Hello"},
    {"type": "token", "content": " world"},
    {"type": "done", "content": "Hello world", "cited_papers": []},
]


async def _fake_astream_chat(*args, **kwargs):
    for ev in MOCK_STREAM_EVENTS:
        yield ev


@pytest.mark.asyncio
async def test_post_message_persists_user_message_before_streaming_response(
    client, session_factory
):
    from sqlalchemy import select

    from backend.db.models import Message

    resp = await client.post("/sessions", json={"mode": "ask"})
    sid = resp.json()["session_id"]

    with patch(
        "backend.api.routers.sessions.astream_chat",
        side_effect=_fake_astream_chat,
    ):
        async with client.stream(
            "POST",
            f"/sessions/{sid}/messages",
            json={"content": "What is ML?"},
        ) as streaming_resp:
            assert streaming_resp.status_code == 200
            async for _ in streaming_resp.aiter_bytes():
                pass

    async with session_factory() as db:
        result = await db.execute(
            select(Message).where(
                Message.session_id == sid, Message.role == "user"
            )
        )
        msgs = result.scalars().all()

    assert len(msgs) == 1
    assert msgs[0].content == "What is ML?"


@pytest.mark.asyncio
async def test_post_message_passes_prior_messages_as_chat_history_to_graph(
    client,
):
    resp = await client.post("/sessions", json={"mode": "ask"})
    sid = resp.json()["session_id"]

    captured_kwargs: dict = {}

    async def capturing_stream(*args, **kwargs):
        captured_kwargs.update(kwargs)
        for ev in MOCK_STREAM_EVENTS:
            yield ev

    with patch(
        "backend.api.routers.sessions.astream_chat",
        side_effect=capturing_stream,
    ):
        # First message
        async with client.stream(
            "POST",
            f"/sessions/{sid}/messages",
            json={"content": "first"},
        ) as r:
            async for _ in r.aiter_bytes():
                pass

        # Second message — chat_history should contain prior exchange
        async with client.stream(
            "POST",
            f"/sessions/{sid}/messages",
            json={"content": "second"},
        ) as r:
            async for _ in r.aiter_bytes():
                pass

    history = captured_kwargs.get("chat_history", [])
    assert len(history) >= 2  # user + assistant from first round


@pytest.mark.asyncio
async def test_post_message_streams_sse_events(client):
    resp = await client.post("/sessions", json={"mode": "ask"})
    sid = resp.json()["session_id"]

    with patch(
        "backend.api.routers.sessions.astream_chat",
        side_effect=_fake_astream_chat,
    ):
        async with client.stream(
            "POST", f"/sessions/{sid}/messages", json={"content": "hi"}
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            events = await collect_sse(r)

    event_types = [e["event"] for e in events]
    assert "token" in event_types
    assert "done" in event_types


@pytest.mark.asyncio
async def test_post_message_persists_assistant_message_on_done(
    client, session_factory
):
    from sqlalchemy import select

    from backend.db.models import Message

    resp = await client.post("/sessions", json={"mode": "ask"})
    sid = resp.json()["session_id"]

    with patch(
        "backend.api.routers.sessions.astream_chat",
        side_effect=_fake_astream_chat,
    ):
        async with client.stream(
            "POST", f"/sessions/{sid}/messages", json={"content": "hi"}
        ) as r:
            async for _ in r.aiter_bytes():
                pass

    async with session_factory() as db:
        result = await db.execute(
            select(Message).where(
                Message.session_id == sid, Message.role == "assistant"
            )
        )
        msgs = result.scalars().all()

    assert len(msgs) == 1
    assert msgs[0].content == "Hello world"


@pytest.mark.asyncio
async def test_post_message_persists_cited_papers_on_citation_event(
    client, session_factory
):
    from sqlalchemy import select

    from backend.db.models import CitedPaper

    citation_events = [
        {"type": "citation", "arxiv_id": "2301.12345", "title": "Some Paper"},
        {"type": "done", "content": "answer", "cited_papers": ["2301.12345"]},
    ]

    async def stream_with_citation(*args, **kwargs):
        for ev in citation_events:
            yield ev

    resp = await client.post("/sessions", json={"mode": "ask"})
    sid = resp.json()["session_id"]

    with patch(
        "backend.api.routers.sessions.astream_chat",
        side_effect=stream_with_citation,
    ):
        async with client.stream(
            "POST", f"/sessions/{sid}/messages", json={"content": "cite me"}
        ) as r:
            async for _ in r.aiter_bytes():
                pass

    async with session_factory() as db:
        result = await db.execute(
            select(CitedPaper).where(CitedPaper.session_id == sid)
        )
        rows = result.scalars().all()

    assert len(rows) == 1
    assert rows[0].arxiv_id == "2301.12345"


@pytest.mark.asyncio
async def test_post_message_for_unknown_session_returns_404(client):
    resp = await client.post(
        f"/sessions/{uuid.uuid4()}/messages", json={"content": "hi"}
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}/messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_messages_returns_history_in_order(client):
    resp = await client.post("/sessions", json={"mode": "ask"})
    sid = resp.json()["session_id"]

    with patch(
        "backend.api.routers.sessions.astream_chat",
        side_effect=_fake_astream_chat,
    ):
        async with client.stream(
            "POST", f"/sessions/{sid}/messages", json={"content": "first question"}
        ) as r:
            async for _ in r.aiter_bytes():
                pass

    resp = await client.get(f"/sessions/{sid}/messages")
    assert resp.status_code == 200
    msgs = resp.json()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# SSE disconnect handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_connection_handles_client_disconnect_gracefully(client):
    """Server must not crash when the client disconnects mid-stream."""

    async def slow_stream(*args, **kwargs):
        yield {"type": "token", "content": "partial"}
        raise GeneratorExit  # simulate abrupt client disconnect

    resp = await client.post("/sessions", json={"mode": "ask"})
    sid = resp.json()["session_id"]

    with patch(
        "backend.api.routers.sessions.astream_chat",
        side_effect=slow_stream,
    ):
        try:
            async with client.stream(
                "POST", f"/sessions/{sid}/messages", json={"content": "hi"}
            ) as r:
                async for chunk in r.aiter_bytes():
                    break  # stop reading after first chunk
        except Exception:
            pass  # client-side abort is expected


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cors_allows_frontend_origin(client):
    resp = await client.options(
        "/papers",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code in (200, 204)
    assert "access-control-allow-origin" in resp.headers


# ---------------------------------------------------------------------------
# Integration tests (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_request_cycle_ask_against_real_graph(client):
    resp = await client.post("/sessions", json={"mode": "ask"})
    sid = resp.json()["session_id"]
    async with client.stream(
        "POST", f"/sessions/{sid}/messages", json={"content": "What is attention?"}
    ) as r:
        events = await collect_sse(r)
    assert any(e["event"] == "done" for e in events)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_request_cycle_deep_dive_against_real_graph(client):
    with patch("backend.api.routers.papers.ingest_paper"):
        pr = await client.post("/papers", json={"arxiv_id": "1706.03762"})
    assert pr.status_code == 202

    resp = await client.post(
        "/sessions", json={"mode": "deep_dive", "paper_id": "1706.03762"}
    )
    sid = resp.json()["session_id"]
    async with client.stream(
        "POST",
        f"/sessions/{sid}/messages",
        json={"content": "Summarise this paper."},
    ) as r:
        events = await collect_sse(r)
    assert any(e["event"] == "done" for e in events)
