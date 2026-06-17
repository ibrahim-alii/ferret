"""Module 1 stub — paper ingestion."""

from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Paper


async def ingest_paper(arxiv_id: str, db: AsyncSession) -> None:
    """Download, chunk, embed, and upsert a paper into Qdrant.

    Updates the paper row's ingestion_status to 'full', 'abstract_only',
    or 'failed' when done.  This stub only sets status to 'full'.
    """
    paper = await db.get(Paper, arxiv_id)
    if paper is None:
        return
    paper.ingestion_status = "full"
    await db.commit()
