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
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

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


async def ensure_collection(client: AsyncQdrantClient) -> None:
    name = _collection_name()
    if await client.collection_exists(collection_name=name):
        return
    await client.create_collection(
        collection_name=name,
        vectors_config={
            _DENSE_NAME: VectorParams(size=_embed_dim(), distance=Distance.COSINE),
        },
        sparse_vectors_config={
            _SPARSE_NAME: SparseVectorParams(index=SparseIndexParams()),
        },
    )


async def upsert_chunks(client: AsyncQdrantClient, chunks: list[ChunkVector]) -> None:
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


async def hybrid_search(
    client: AsyncQdrantClient,
    query_dense: list[float],
    query_text: str,
    paper_id: str | None,
    limit: int,
) -> list[ScoredChunk]:
    name = _collection_name()
    sparse_vec = await asyncio.to_thread(encode_query, query_text)

    qfilter: Filter | None = None
    if paper_id is not None:
        qfilter = Filter(
            must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
        )

    prefetch = [
        Prefetch(query=query_dense, using=_DENSE_NAME, limit=limit * 2),
        Prefetch(
            query=SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
            using=_SPARSE_NAME,
            limit=limit * 2,
        ),
    ]

    response = await client.query_points(
        collection_name=name,
        prefetch=prefetch,
        query=FusionQuery(fusion=Fusion.RRF),
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
