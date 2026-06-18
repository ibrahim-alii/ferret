import asyncio
import logging
import os

import voyageai

from backend.vectorstore.store import hybrid_search

logger = logging.getLogger(__name__)


async def retrieve_node(state: dict) -> dict:
    embed_model = os.environ.get("VOYAGE_EMBED_MODEL", "voyage-4-lite")
    limit = int(os.environ.get("RERANK_CANDIDATE_COUNT", "30"))

    max_concurrency = int(os.environ.get("VOYAGE_MAX_CONCURRENCY", "2"))
    _semaphore = asyncio.Semaphore(max_concurrency)

    client = voyageai.AsyncClient()
    async with _semaphore:
        result = await client.embed(
            [state["user_message"]],
            model=embed_model,
        )
    query_dense = result.embeddings[0]

    paper_id = state["paper_id"] if state["mode"] == "deep_dive" else None

    chunks = await hybrid_search(
        query_dense=query_dense,
        query_text=state["user_message"],
        paper_id=paper_id,
        limit=limit,
    )

    return {"retrieved_chunks": chunks}
