"""
Ingestion entrypoint.
ingest_paper(arxiv_id) -> PaperRecord is the single public function for this module.
It is idempotent: a second call on an already-'full' paper is a no-op.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from sqlalchemy import select

from backend.db.models import Chunk, Paper
from backend.db.session import async_session
from backend.ingestion.arxiv_client import download_pdf, fetch_html, fetch_metadata
from backend.ingestion.chunker import chunk_sections
from backend.ingestion.embedder import embed_chunks
from backend.ingestion.filter import filter_sections
from backend.ingestion.parser import parse
from backend.vectorstore.interface import ChunkVector, upsert_chunks

log = logging.getLogger(__name__)


@dataclass
class PaperRecord:
    arxiv_id: str
    title: str
    abstract: str
    ingestion_status: str


async def ingest_paper(arxiv_id: str) -> PaperRecord:
    """
    Fetch, parse, chunk, embed, and store an arxiv paper.
    Idempotent: returns existing record immediately if ingestion_status == 'full'.
    """
    async with async_session() as session:
        result = await session.execute(select(Paper).where(Paper.arxiv_id == arxiv_id))
        existing = result.scalar_one_or_none()

        if existing is not None and existing.ingestion_status == "full":
            log.debug("Paper %s already fully ingested; skipping", arxiv_id)
            return PaperRecord(
                arxiv_id=existing.arxiv_id,
                title=existing.title,
                abstract=existing.abstract,
                ingestion_status=existing.ingestion_status,
            )

        # Concurrent fetch: metadata + HTML/PDF in parallel
        meta, (html_content, pdf_bytes) = await asyncio.gather(
            fetch_metadata(arxiv_id),
            _fetch_content(arxiv_id),
        )

        title = meta["title"]
        abstract = meta["abstract"]
        authors = meta["authors"]
        published_date = meta["published_date"]

        raw_sections = parse(html_content=html_content, pdf_bytes=pdf_bytes)
        filtered_sections = filter_sections(raw_sections)
        parents, children = chunk_sections(
            paper_id=0,  # placeholder; FK set after paper flush
            abstract=abstract,
            sections=filtered_sections,
        )

        status = "full" if filtered_sections else "abstract_only"

        if existing is None:
            paper = Paper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=json.dumps(authors),
                published_date=published_date,
                ingestion_status="pending",
            )
            session.add(paper)
        else:
            paper = existing
            paper.title = title
            paper.abstract = abstract
            paper.authors = json.dumps(authors)
            paper.published_date = published_date

        await session.flush()  # assigns paper.id

        for chunk in parents + children:
            chunk.paper_id = paper.id

        for parent in parents:
            session.add(parent)
        await session.flush()  # assigns parent IDs

        for child in children:
            parent_ref = getattr(child, "_parent_ref", None)
            if parent_ref is not None:
                child.parent_chunk_id = parent_ref.id
            session.add(child)
        await session.flush()

        # Embed and upsert child chunks only (parents never embedded)
        if children and status == "full":
            vectors = await embed_chunks(children)
            chunk_vectors: list[ChunkVector] = [
                ChunkVector(
                    chunk_id=child.id,
                    paper_id=paper.id,
                    section_name=child.section_name,
                    chunk_type=child.chunk_type,
                    dense_vector=vector,
                    text=child.text,
                )
                for child, vector in zip(children, vectors)
            ]
            await upsert_chunks(chunk_vectors)

        paper.ingestion_status = status
        await session.commit()

        return PaperRecord(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            abstract=paper.abstract,
            ingestion_status=paper.ingestion_status,
        )


async def _fetch_content(arxiv_id: str) -> tuple[str | None, bytes | None]:
    """
    Fetch paper content.
    ARXIV_PREFER_HTML=true (default): fetch HTML and PDF concurrently; parser tries HTML first.
    ARXIV_PREFER_HTML=false: skip HTML, fetch PDF only (for older papers or PDF-only submissions).
    Each fetch soft-fails independently.
    """
    import os as _os
    prefer_html = _os.environ.get("ARXIV_PREFER_HTML", "true").lower() not in ("false", "0", "no")

    html_content: str | None = None
    pdf_bytes: bytes | None = None

    if prefer_html:
        html_task = asyncio.create_task(fetch_html(arxiv_id))
        pdf_task = asyncio.create_task(download_pdf(arxiv_id))

        try:
            html_content = await html_task
        except Exception as exc:
            log.warning("HTML fetch failed for %s: %s", arxiv_id, exc)

        try:
            pdf_bytes = await pdf_task
        except Exception as exc:
            log.warning("PDF fetch failed for %s: %s", arxiv_id, exc)
    else:
        try:
            pdf_bytes = await download_pdf(arxiv_id)
        except Exception as exc:
            log.warning("PDF fetch failed for %s: %s", arxiv_id, exc)

    return html_content, pdf_bytes
