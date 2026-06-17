from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkVector:
    chunk_id: str
    paper_id: str
    section_name: str
    chunk_type: str
    dense_vector: tuple[float, ...]
    text: str


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk_id: str
    paper_id: str
    section_name: str
    chunk_type: str
    score: float
