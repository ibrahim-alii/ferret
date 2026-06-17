"""Pydantic request/response schemas for all endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------


class PostPaperRequest(BaseModel):
    arxiv_id: str

    @field_validator("arxiv_id")
    @classmethod
    def arxiv_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("arxiv_id must not be empty")
        return v.strip()


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
    def deep_dive_requires_paper_id(self) -> "PostSessionRequest":
        if self.mode == "deep_dive" and not self.paper_id:
            raise ValueError("paper_id is required for deep_dive mode")
        return self


class PostSessionResponse(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class PostMessageRequest(BaseModel):
    content: str


class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}
