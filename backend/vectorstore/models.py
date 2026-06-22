from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkVector:
    chunk_id: int  # SQLite chunks.id (autoincrement); also the Qdrant point id
    paper_id: str
    section_name: str
    chunk_type: str
    dense_vector: list[float]
    text: str
    content_type: str = "text"  # "text" | "table" | "figure"


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk_id: str
    paper_id: str
    section_name: str
    chunk_type: str
    score: float
    content_type: str = "text"
    # Dense embedding, populated only when the query requests vectors (MMR de-dup).
    dense_vector: list[float] | None = None
