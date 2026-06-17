"""Module 1 stub — paper ingestion."""

from backend.db.models import Paper
from backend.db.session import session_context


async def ingest_paper(arxiv_id: str) -> None:
    """Download, chunk, embed, and upsert a paper into Qdrant.

    Owns its own DB session so it can safely run as a FastAPI BackgroundTask
    after the request session has been closed.
    """
    async with session_context() as db:
        paper = await db.get(Paper, arxiv_id)
        if paper is None:
            return
        paper.ingestion_status = "full"
        await db.commit()
