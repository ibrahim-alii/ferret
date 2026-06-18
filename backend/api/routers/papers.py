"""POST /papers and GET /papers/{arxiv_id} endpoints."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.schemas import PaperStatusResponse, PostPaperRequest
from backend.db.models import Paper
from backend.db.session import get_session
from backend.ingestion.ingest import ingest_paper

router = APIRouter(prefix="/papers", tags=["papers"])


@router.post("", status_code=202, response_model=PaperStatusResponse)
async def post_paper(
    body: PostPaperRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session),
) -> PaperStatusResponse:
    paper = await db.get(Paper, body.arxiv_id)
    if paper is None:
        paper = Paper(arxiv_id=body.arxiv_id, ingestion_status="pending")
        db.add(paper)
        await db.commit()
        await db.refresh(paper)

    # Queue ingestion — ingest_paper owns its own session to avoid using the
    # request-scoped session after it is closed.
    background_tasks.add_task(ingest_paper, body.arxiv_id)

    return PaperStatusResponse(
        arxiv_id=paper.arxiv_id,
        ingestion_status=paper.ingestion_status,
        title=paper.title,
        abstract=paper.abstract,
    )


@router.get("/{arxiv_id}", response_model=PaperStatusResponse)
async def get_paper(
    arxiv_id: str,
    db: AsyncSession = Depends(get_session),
) -> PaperStatusResponse:
    paper = await db.get(Paper, arxiv_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return PaperStatusResponse(
        arxiv_id=paper.arxiv_id,
        ingestion_status=paper.ingestion_status,
        title=paper.title,
        abstract=paper.abstract,
    )
