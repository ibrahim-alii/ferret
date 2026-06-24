"""Unit tests for the Gemini embedding rate limiter (token bucket)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.ingestion import embedder


@pytest.mark.asyncio
async def test_limiter_immediate_when_tokens_available(monkeypatch):
    monkeypatch.setattr(embedder.time, "monotonic", lambda: 0.0)
    slept = []
    async def fake_sleep(d):
        slept.append(d)
    monkeypatch.setattr(embedder.asyncio, "sleep", fake_sleep)

    lim = embedder._AsyncRateLimiter(rate_per_min=90, capacity=100)
    await lim.acquire(50)  # bucket starts full -> no wait

    assert slept == []
    assert lim.throttled is False


@pytest.mark.asyncio
async def test_limiter_paces_and_flags_throttle_when_exhausted(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(embedder.time, "monotonic", lambda: clock[0])
    sleeps = []
    async def fake_sleep(d):
        sleeps.append(d)
        clock[0] += d  # advance the clock so tokens refill
    monkeypatch.setattr(embedder.asyncio, "sleep", fake_sleep)

    lim = embedder._AsyncRateLimiter(rate_per_min=60, capacity=10)  # 1 token/sec
    await lim.acquire(10)  # drains the bucket, still no wait
    assert sleeps == []
    assert lim.throttled is False

    await lim.acquire(5)  # needs 5 tokens, none left -> waits 5s, then refills
    assert lim.throttled is True
    assert sleeps == [5.0]


@pytest.mark.asyncio
async def test_acquire_clamps_request_to_capacity(monkeypatch):
    # n larger than capacity must not deadlock; it's clamped to capacity.
    clock = [0.0]
    monkeypatch.setattr(embedder.time, "monotonic", lambda: clock[0])
    async def fake_sleep(d):
        clock[0] += d
    monkeypatch.setattr(embedder.asyncio, "sleep", fake_sleep)

    lim = embedder._AsyncRateLimiter(rate_per_min=60, capacity=10)
    await lim.acquire(10)
    await lim.acquire(1000)  # clamped to 10; returns after refill instead of hanging


@pytest.mark.asyncio
async def test_embed_batch_gemini_acquires_one_token_per_text(monkeypatch):
    monkeypatch.setenv("USE_LOCAL_EMBEDDINGS", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    acquired = []
    class StubLimiter:
        throttled = False
        async def acquire(self, n=1.0):
            acquired.append(n)
    monkeypatch.setattr(embedder, "_gemini_limiter", StubLimiter())

    emb = MagicMock(values=[0.1, 0.2])
    client = MagicMock()
    client.aio.models.embed_content = AsyncMock(return_value=MagicMock(embeddings=[emb, emb]))

    out = await embedder._embed_batch(client, ["a", "b"], asyncio.Semaphore(1), is_query=False)

    assert acquired == [2]  # one token per chunk
    assert out == [[0.1, 0.2], [0.1, 0.2]]
