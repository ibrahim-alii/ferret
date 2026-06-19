"""Collection management, upsert, and hybrid search."""
from __future__ import annotations

import asyncio
import logging
import os
from functools import cache
from itertools import islice
from typing import Iterable, Iterator, TypeVar

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from backend.vectorstore.client import get_client
from backend.vectorstore.models import ChunkVector, ScoredChunk
from backend.vectorstore.sparse import encode_document, encode_query

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100
_DENSE_NAME = "dense"
_SPARSE_NAME = "sparse"

T = TypeVar("T")


@cache
def _collection_name() -> str:
    return os.environ.get("QDRANT_COLLECTION_NAME", "arxiv_chunks")


@cache
def _embed_dim() -> int:
    return int(os.environ.get("VOYAGE_EMBED_DIM", "1024"))


def _batched(iterable: Iterable[T], n: int) -> Iterator[list[T]]:
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


# ---------------------------------------------------------------------------
# Internal implementations (accept an injected client — used by unit tests)
# ---------------------------------------------------------------------------


async def _ensure_collection(client: AsyncQdrantClient) -> None:
    """Create the Qdrant collection with dense + sparse named vectors, idempotently."""
    name = _collection_name()
    if not await client.collection_exists(collection_name=name):
        await client.create_collection(
            collection_name=name,
            vectors_config={
                _DENSE_NAME: VectorParams(size=_embed_dim(), distance=Distance.COSINE),
            },
            sparse_vectors_config={
                _SPARSE_NAME: SparseVectorParams(index=SparseIndexParams()),
            },
        )
    # Payload index on paper_id is required for Deep Dive's filtered queries. Idempotent.
    await client.create_payload_index(
        collection_name=name,
        field_name="paper_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )


async def _upsert_chunks(client: AsyncQdrantClient, chunks: list[ChunkVector]) -> None:
    name = _collection_name()
    for batch in _batched(chunks, _BATCH_SIZE):
        points: list[PointStruct] = []
        for chunk in batch:
            sparse_vec = await asyncio.to_thread(encode_document, chunk.text)
            point = PointStruct(
                id=chunk.chunk_id,
                vector={
                    _DENSE_NAME: chunk.dense_vector,
                    _SPARSE_NAME: SparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
                    ),
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "section_name": chunk.section_name,
                    "chunk_type": chunk.chunk_type,
                },
            )
            points.append(point)
        try:
            await client.upsert(collection_name=name, points=points)
        except Exception:
            logger.exception(
                "Upsert failed for batch of %d points in collection %r", len(points), name
            )
            raise


async def _hybrid_search(
    client: AsyncQdrantClient,
    query_dense: list[float],
    query_text: str,
    paper_id: str | None,
    limit: int,
    use_sparse: bool = True,
) -> list[ScoredChunk]:
    name = _collection_name()

    qfilter: Filter | None = None
    if paper_id is not None:
        qfilter = Filter(
            must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
        )

    if use_sparse:
        sparse_vec = await asyncio.to_thread(encode_query, query_text)
        prefetch: list[Prefetch] = [
            Prefetch(query=query_dense, using=_DENSE_NAME, limit=limit * 2),
            Prefetch(
                query=SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                using=_SPARSE_NAME,
                limit=limit * 2,
            ),
        ]
        query_arg: FusionQuery | list[float] = FusionQuery(fusion=Fusion.RRF)
        prefetch_arg: list[Prefetch] | None = prefetch
        using_arg: str | None = None
    else:
        # Dense-only: issue a direct ANN query; no prefetch or RRF needed.
        prefetch_arg = None
        query_arg = query_dense
        using_arg = _DENSE_NAME

    response = await client.query_points(
        collection_name=name,
        prefetch=prefetch_arg,
        query=query_arg,
        using=using_arg,
        limit=limit,
        query_filter=qfilter,
        with_payload=True,
    )

    results: list[ScoredChunk] = []
    for hit in response.points:
        payload = hit.payload or {}
        paper = payload.get("paper_id", "")
        if not paper:
            logger.warning("Point %s missing paper_id in payload", hit.id)
        results.append(
            ScoredChunk(
                chunk_id=payload.get("chunk_id", str(hit.id)),
                paper_id=paper,
                section_name=payload.get("section_name", ""),
                chunk_type=payload.get("chunk_type", ""),
                score=hit.score,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Public API (as per PLAN.md contract — no client parameter)
# ---------------------------------------------------------------------------


async def ensure_collection() -> None:
    """Create the Qdrant collection idempotently. Consumed by `ferret init`."""
    async with get_client() as client:
        await _ensure_collection(client)


async def upsert_chunks(chunks: list[ChunkVector]) -> None:
    """Batch-upsert chunks into Qdrant. Consumed by module 1 (ingestion)."""
    async with get_client() as client:
        await _upsert_chunks(client, chunks)


async def hybrid_search(
    query_dense: list[float],
    query_text: str,
    paper_id: str | None,
    limit: int,
    use_sparse: bool = True,
) -> list[ScoredChunk]:
    """Server-side RRF hybrid search. Consumed by module 3 (retrieval).

    paper_id=None -> Ask mode (search all papers).
    paper_id=<id> -> Deep Dive mode (filter to one paper).
    use_sparse=False -> dense-only (for eval comparison).
    """
    async with get_client() as client:
        return await _hybrid_search(client, query_dense, query_text, paper_id, limit, use_sparse)
