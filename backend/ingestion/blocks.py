"""Parsed-block contract shared across parser -> filter -> chunker.

A block is one unit of paper content: a prose passage, a table serialized to
GitHub-flavored markdown, or a figure (caption + image reference). content_type
threads through chunking into the Chunk rows and Qdrant payload so retrieval and
generation can treat tables/figures specially (atomic chunks, rendered output).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    section_name: str
    text: str  # prose, GFM table markdown, or figure caption
    content_type: str = "text"  # "text" | "table" | "figure"
    media_url: str | None = None  # figures only: arxiv.org URL or "/media/<id>/<file>"
