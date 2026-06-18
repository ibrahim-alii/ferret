from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    arxiv_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    abstract: Mapped[str] = mapped_column(Text, nullable=False, default="")
    authors: Mapped[str] = mapped_column(Text, nullable=False, default="[]")  # JSON array
    published_date: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    ingestion_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending"
    )  # "pending" | "abstract_only" | "full"
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list[Chunk]] = relationship(
        "Chunk", back_populates="paper", cascade="all, delete-orphan"
    )

    def get_authors(self) -> list[str]:
        try:
            return json.loads(self.authors)
        except (json.JSONDecodeError, TypeError):
            import logging
            logging.getLogger(__name__).warning("Malformed authors JSON for paper %s", self.arxiv_id)
            return []

    def set_authors(self, authors: list[str]) -> None:
        self.authors = json.dumps(authors)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_type: Mapped[str] = mapped_column(String(16), nullable=False)  # "parent" | "child"
    section_name: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    parent_chunk_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True
    )
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    paper: Mapped[Paper] = relationship("Paper", back_populates="chunks")

    __table_args__ = (Index("ix_chunks_paper_type", "paper_id", "chunk_type"),)


class Session(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    mode: Mapped[str] = mapped_column(String)
    paper_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("papers.arxiv_id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    messages: Mapped[list[Message]] = relationship(
        "Message", back_populates="session", order_by="Message.created_at"
    )
    cited_papers: Mapped[list[CitedPaper]] = relationship(
        "CitedPaper", back_populates="session"
    )


class Message(Base):
    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("sessions.session_id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    session: Mapped[Session] = relationship("Session", back_populates="messages")


class CitedPaper(Base):
    __tablename__ = "cited_papers"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("sessions.session_id"), nullable=False
    )
    message_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("messages.message_id"), nullable=True
    )
    arxiv_id: Mapped[str] = mapped_column(String)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)

    session: Mapped[Session] = relationship(
        "Session", back_populates="cited_papers"
    )
