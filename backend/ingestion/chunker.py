"""
Parent/child chunker.
One parent chunk per section (+ one for the abstract/metadata).
Each parent is split into child chunks of CHILD_CHUNK_TOKENS tokens with CHILD_CHUNK_OVERLAP overlap.
Parents are stored in SQLite only; children are the retrieval units sent to the vector store.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import tiktoken

from backend.db.models import Chunk
from backend.ingestion.blocks import ParsedBlock

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_split(text: str, max_tokens: int, overlap: int) -> list[str]:
    tokens = _ENCODING.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(_ENCODING.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += max_tokens - overlap
    return chunks if chunks else [text]


def chunk_sections(
    paper_id: int,
    abstract: str,
    sections: list[ParsedBlock],
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Returns (parents, children).
    Parents: one Chunk per block + one Abstract parent; chunk_type="parent".
    Children: text-split from each parent; chunk_type="child".
    content_type/media_url propagate to both parent and child rows.
    Table/figure blocks are ATOMIC — one parent + one child, never token-split, so
    markdown tables stay intact and figure captions keep their image reference.
    child._parent_ref holds the parent Chunk so ingest.py can wire parent_chunk_id after DB flush.
    """
    max_tokens = int(os.environ.get("CHILD_CHUNK_TOKENS", "512"))
    overlap = int(os.environ.get("CHILD_CHUNK_OVERLAP", "64"))
    now = datetime.now(tz=timezone.utc)

    parents: list[Chunk] = []
    children: list[Chunk] = []

    abstract_parent = Chunk(
        paper_id=paper_id,
        chunk_type="parent",
        section_name="Abstract",
        text=abstract,
        content_type="text",
        media_url=None,
        parent_chunk_id=None,
        token_count=len(_ENCODING.encode(abstract)),
        created_at=now,
    )
    parents.append(abstract_parent)

    for block in sections:
        parent = Chunk(
            paper_id=paper_id,
            chunk_type="parent",
            section_name=block.section_name,
            text=block.text,
            content_type=block.content_type,
            media_url=block.media_url,
            parent_chunk_id=None,
            token_count=len(_ENCODING.encode(block.text)),
            created_at=now,
        )
        parents.append(parent)

        # Tables/figures are atomic; only prose is sliding-window split.
        if block.content_type == "text":
            child_texts = _token_split(block.text, max_tokens, overlap)
        else:
            child_texts = [block.text]

        for child_text in child_texts:
            child = Chunk(
                paper_id=paper_id,
                chunk_type="child",
                section_name=block.section_name,
                text=child_text,
                content_type=block.content_type,
                media_url=block.media_url,
                parent_chunk_id=None,  # wired after DB flush in ingest.py
                token_count=len(_ENCODING.encode(child_text)),
                created_at=now,
            )
            child._parent_ref = parent  # type: ignore[attr-defined]
            children.append(child)

    return parents, children
