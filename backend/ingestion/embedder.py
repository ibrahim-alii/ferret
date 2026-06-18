"""
Voyage AI embedding wrapper for child chunks.
Batches requests, bounds concurrency via asyncio.Semaphore, retries on transient errors.
"""
from __future__ import annotations

import asyncio
import logging
import os

import voyageai

from backend.db.models import Chunk

log = logging.getLogger(__name__)

_MAX_RETRIES = 3

_DEFAULT_MODEL = "voyage-4-lite"
_DEFAULT_BATCH_SIZE = 128
_DEFAULT_MAX_CONCURRENCY = 5


async def _embed_batch(
    client: voyageai.AsyncClient,
    texts: list[str],
    semaphore: asyncio.Semaphore,
    model: str,
) -> list[list[float]]:
    for attempt in range(_MAX_RETRIES):
        try:
            async with semaphore:
                result = await client.embed(texts, model=model, input_type="document")
            return result.embeddings
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = 2**attempt
            log.warning(
                "Voyage embed attempt %d failed (%s); retrying in %ds",
                attempt + 1,
                exc,
                wait,
            )
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


async def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """
    Embed a list of chunks using Voyage AI dense embeddings.
    Returns vectors in the same order as input chunks.
    """
    batch_size = int(os.environ.get("VOYAGE_BATCH_SIZE", str(_DEFAULT_BATCH_SIZE)))
    max_concurrency = int(os.environ.get("VOYAGE_MAX_CONCURRENCY", str(_DEFAULT_MAX_CONCURRENCY)))
    model = os.environ.get("VOYAGE_MODEL", _DEFAULT_MODEL)

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise EnvironmentError("VOYAGE_API_KEY environment variable is not set")
    client = voyageai.AsyncClient(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrency)

    batches = [
        [c.text for c in chunks[i : i + batch_size]]
        for i in range(0, len(chunks), batch_size)
    ]

    tasks = [_embed_batch(client, batch, semaphore, model) for batch in batches]
    results = await asyncio.gather(*tasks)

    vectors: list[list[float]] = []
    for batch_vectors in results:
        vectors.extend(batch_vectors)
    return vectors
