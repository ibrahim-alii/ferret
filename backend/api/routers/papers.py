"""POST /papers and GET /papers/{arxiv_id} endpoints."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.limiter import limiter
from backend.api.schemas import PaperStatusResponse, PostPaperRequest
from backend.db.models import Paper
from backend.db.session import get_session
from backend.ingestion.ingest import ingest_paper

router = APIRouter(prefix="/papers", tags=["papers"])


async def _get_paper_by_arxiv_id(db: AsyncSession, arxiv_id: str) -> Paper | None:
    result = await db.execute(select(Paper).where(Paper.arxiv_id == arxiv_id))
    return result.scalar_one_or_none()


@router.post("", status_code=202, response_model=PaperStatusResponse)
@limiter.limit("10/minute")
async def post_paper(
    request: Request,
    body: PostPaperRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session),
) -> PaperStatusResponse:
    paper = await _get_paper_by_arxiv_id(db, body.arxiv_id)
    if paper is None:
        paper = Paper(arxiv_id=body.arxiv_id, ingestion_status="pending")
        db.add(paper)
        await db.commit()
        await db.refresh(paper)

    # Queue ingestion unless already fully ingested; "abstract_only" (partial
    # parse failure) should also re-trigger so callers can retry degraded papers.
    # ingest_paper owns its own session to avoid using the request-scoped one.
    if paper.ingestion_status != "full":
        background_tasks.add_task(ingest_paper, body.arxiv_id)

    return PaperStatusResponse(
        arxiv_id=paper.arxiv_id,
        ingestion_status=paper.ingestion_status,
        title=paper.title,
        abstract=paper.abstract,
    )


@router.get("/{arxiv_id}", response_model=PaperStatusResponse)
@limiter.limit("60/minute")
async def get_paper(
    request: Request,
    arxiv_id: str,
    db: AsyncSession = Depends(get_session),
) -> PaperStatusResponse:
    paper = await _get_paper_by_arxiv_id(db, arxiv_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return PaperStatusResponse(
        arxiv_id=paper.arxiv_id,
        ingestion_status=paper.ingestion_status,
        title=paper.title,
        abstract=paper.abstract,
    )
