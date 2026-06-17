"""
Stub for Module 2's Qdrant upsert interface.
Module 2 owns the real implementation; this file defines the contract only.
"""
from __future__ import annotations

from typing import TypedDict


class ChunkVector(TypedDict):
    chunk_id: int
    paper_id: int
    section_name: str
    chunk_type: str
    dense_vector: list[float]
    text: str


async def upsert_chunks(chunks: list[ChunkVector]) -> None:
    raise NotImplementedError("Module 2 owns this implementation")
