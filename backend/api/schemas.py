"""Pydantic request/response schemas for all endpoints."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

_ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------


class PostPaperRequest(BaseModel):
    arxiv_id: str

    @field_validator("arxiv_id")
    @classmethod
    def arxiv_id_valid(cls, v: str) -> str:
        v = v.strip()
        if not _ARXIV_RE.match(v):
            raise ValueError("Invalid arXiv ID format, expected e.g. 2301.00001")
        return v


class PaperStatusResponse(BaseModel):
    arxiv_id: str
    ingestion_status: str
    title: str | None = None
    abstract: str | None = None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class PostSessionRequest(BaseModel):
    mode: Literal["ask", "deep_dive"]
    paper_id: str | None = None

    @model_validator(mode="after")
    def deep_dive_requires_paper_id(self) -> PostSessionRequest:
        if self.mode == "deep_dive" and not self.paper_id:
            raise ValueError("paper_id is required for deep_dive mode")
        return self


class PostSessionResponse(BaseModel):
    session_id: str


class SessionSummary(BaseModel):
    session_id: str
    mode: str
    paper_id: str | None = None
    created_at: datetime
    title: str


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class PostMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}
