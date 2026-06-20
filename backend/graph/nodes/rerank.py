import asyncio
import logging
import os

import voyageai

from backend.graph.stream import emit

logger = logging.getLogger(__name__)

# Shared across requests so VOYAGE_MAX_CONCURRENCY bounds total in-flight Voyage calls.
_semaphore = asyncio.Semaphore(int(os.environ.get("VOYAGE_MAX_CONCURRENCY", "2")))


async def rerank_node(state: dict) -> dict:
    emit({"type": "status", "step": "reranking", "content": "Ranking results"})
    rerank_model = os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2.5-lite")
    top_k = int(os.environ.get("RERANK_TOP_K", "5"))

    chunks = state["retrieved_chunks"]
    if not chunks:
        return {"reranked_children": []}

    query = state["user_message"]
    documents = [c.get("text", "") for c in chunks]

    client = voyageai.AsyncClient()
    async with _semaphore:
        result = await client.rerank(
            query=query,
            documents=documents,
            model=rerank_model,
            top_k=top_k,
        )

    reranked = []
    for r in result.results:
        chunk = dict(chunks[r.index])
        chunk["score"] = r.relevance_score
        reranked.append(chunk)

    # Sort by score descending, take top_k
    reranked.sort(key=lambda x: x["score"], reverse=True)
    return {"reranked_children": reranked[:top_k]}
