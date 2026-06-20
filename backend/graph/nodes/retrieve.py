import logging
import os

import aiosqlite

from backend.graph.stream import emit
from backend.ingestion.embedder import embed_query
from backend.vectorstore.models import ScoredChunk
from backend.vectorstore.store import hybrid_search

logger = logging.getLogger(__name__)


async def _enrich_with_text(scored: list[ScoredChunk]) -> list[dict]:
    """Attach child text and parent_chunk_id (kept in SQLite, not Qdrant) to each hit.

    Children are the retrieval unit; their text is needed for reranking and their
    parent_chunk_id for small-to-big expansion. Qdrant stores neither (PRD Section 3),
    so we look them up by chunk id in one query.
    """
    if not scored:
        return []
    # Already-enriched dicts (carry their own text) pass through unchanged.
    if isinstance(scored[0], dict):
        return list(scored)
    db_path = os.environ.get("SQLITE_DB_PATH", "./backend/data/app.db")
    ids = [c.chunk_id for c in scored]
    placeholders = ",".join("?" * len(ids))
    text_by_id: dict = {}
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(
            f"SELECT id, text, parent_chunk_id FROM chunks WHERE id IN ({placeholders})",
            ids,
        )
        async for row in cursor:
            text_by_id[row["id"]] = (row["text"], row["parent_chunk_id"])

    enriched: list[dict] = []
    for c in scored:
        text, parent_chunk_id = text_by_id.get(c.chunk_id, ("", None))
        enriched.append(
            {
                "chunk_id": c.chunk_id,
                "paper_id": c.paper_id,
                "section_name": c.section_name,
                "chunk_type": c.chunk_type,
                "score": c.score,
                "text": text,
                "parent_chunk_id": parent_chunk_id,
            }
        )
    return enriched


async def retrieve_node(state: dict) -> dict:
    emit({"type": "status", "step": "retrieving", "content": "Searching papers"})
    limit = int(os.environ.get("RERANK_CANDIDATE_COUNT", "30"))

    query_dense = await embed_query(state["user_message"])

    paper_id = state["paper_id"] if state["mode"] == "deep_dive" else None

    chunks = await hybrid_search(
        query_dense=query_dense,
        query_text=state["user_message"],
        paper_id=paper_id,
        limit=limit,
    )

    return {"retrieved_chunks": await _enrich_with_text(chunks)}
