"""
Dense embedding wrapper for child chunks and queries.

Provider is selected by USE_LOCAL_EMBEDDINGS: when true, embeddings come from Google
Gemini (a local/dev testing path that avoids OpenAI quota); otherwise from OpenAI. Both
produce the same dimensionality (OPENAI_EMBED_DIM) so they share one Qdrant collection.
Batches requests, bounds concurrency via asyncio.Semaphore, retries on transient errors.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from google import genai
from google.genai import types
from openai import AsyncOpenAI

from backend.db.models import Chunk

log = logging.getLogger(__name__)

_MAX_RETRIES = 3

_DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
_DEFAULT_GEMINI_MODEL = "gemini-embedding-001"
_DEFAULT_BATCH_SIZE = 128
_DEFAULT_MAX_CONCURRENCY = 5
_GEMINI_MAX_BATCH = 100
_DEFAULT_GEMINI_RPM = 90  # 10% headroom under the free tier's 100 requests/minute


class _AsyncRateLimiter:
    """Token-bucket limiter pacing embed calls to a sustained requests/minute rate.

    Gemini's free embedding tier caps requests-per-minute, and each chunk counts as
    one request, so a large paper (>100 chunks) bursts straight into 429s. We refill
    ``rate_per_min / 60`` tokens per second up to ``capacity`` and make every call
    consume one token per text, so ingestion paces itself under the quota instead of
    failing. ``throttled`` flips True whenever a call actually had to wait, so callers
    can surface that to the user. Reliable, not fast: throughput converges to the rate.
    """

    def __init__(self, rate_per_min: float, capacity: float) -> None:
        self._rate = rate_per_min / 60.0  # tokens per second
        self._capacity = capacity
        self._tokens = capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()
        self.throttled = False

    async def acquire(self, n: float = 1.0) -> None:
        n = min(n, self._capacity)  # never ask for more than the bucket can ever hold
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
                self._updated = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                wait = (n - self._tokens) / self._rate
            self.throttled = True
            log.info("Embedding rate limit reached; pacing ingestion (waiting %.1fs)", wait)
            await asyncio.sleep(wait)


_gemini_rpm = float(os.environ.get("GEMINI_EMBED_RPM", str(_DEFAULT_GEMINI_RPM)))
# Capacity covers a full max-size batch so acquire() can never deadlock on n > capacity.
_gemini_limiter = _AsyncRateLimiter(rate_per_min=_gemini_rpm, capacity=max(_gemini_rpm, _GEMINI_MAX_BATCH))


def _use_gemini() -> bool:
    return os.environ.get("USE_LOCAL_EMBEDDINGS", "false").strip().lower() in ("1", "true", "yes")


def _embed_dim() -> int:
    return int(os.environ.get("OPENAI_EMBED_DIM", "1536"))


def _make_client():
    """Build the async embedding client for the active provider."""
    if _use_gemini():
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set")
        return genai.Client(api_key=api_key)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set")
    return AsyncOpenAI(api_key=api_key)


async def _embed_batch(
    client,
    texts: list[str],
    semaphore: asyncio.Semaphore,
    *,
    is_query: bool,
) -> list[list[float]]:
    for attempt in range(_MAX_RETRIES):
        try:
            async with semaphore:
                if _use_gemini():
                    await _gemini_limiter.acquire(len(texts))
                    model = os.environ.get("GEMINI_EMBED_MODEL", _DEFAULT_GEMINI_MODEL)
                    result = await client.aio.models.embed_content(
                        model=model,
                        contents=texts,
                        config=types.EmbedContentConfig(
                            task_type="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT",
                            output_dimensionality=_embed_dim(),
                        ),
                    )
                    return [e.values for e in result.embeddings]

                model = os.environ.get("OPENAI_EMBED_MODEL", _DEFAULT_OPENAI_MODEL)
                result = await client.embeddings.create(input=texts, model=model)
                return [d.embedding for d in result.data]
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = 2**attempt
            log.warning(
                "Embed attempt %d failed (%s); retrying in %ds",
                attempt + 1,
                exc,
                wait,
            )
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


async def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """
    Embed a list of chunks (documents) using the active provider's dense embeddings.
    Returns vectors in the same order as input chunks.
    """
    batch_size = int(os.environ.get("OPENAI_BATCH_SIZE", str(_DEFAULT_BATCH_SIZE)))
    if _use_gemini():
        # Gemini's BatchEmbedContents rejects any batch over 100 requests.
        batch_size = min(batch_size, _GEMINI_MAX_BATCH)

    client = _make_client()
    # Shared at module level so concurrent ingestions (e.g. Ask's corrective branch
    # ingesting several papers at once) stay within one provider-wide concurrency
    # budget rather than each spawning its own full quota of in-flight calls.
    semaphore = _doc_semaphore

    batches = [
        [c.text for c in chunks[i : i + batch_size]]
        for i in range(0, len(chunks), batch_size)
    ]

    tasks = [_embed_batch(client, batch, semaphore, is_query=False) for batch in batches]
    results = await asyncio.gather(*tasks)

    vectors: list[list[float]] = []
    for batch_vectors in results:
        vectors.extend(batch_vectors)
    return vectors


# Shared across requests so the active provider's concurrency limit bounds total in-flight
# embedding calls, rather than allowing one full quota per concurrent request.
_doc_semaphore = asyncio.Semaphore(
    int(os.environ.get("OPENAI_MAX_CONCURRENCY", str(_DEFAULT_MAX_CONCURRENCY)))
)
_query_semaphore = asyncio.Semaphore(
    int(os.environ.get("OPENAI_MAX_CONCURRENCY", str(_DEFAULT_MAX_CONCURRENCY)))
)


async def embed_query(text: str) -> list[float]:
    """Embed a single user query and return one dense vector."""
    client = _make_client()
    vectors = await _embed_batch(client, [text], _query_semaphore, is_query=True)
    return vectors[0]
