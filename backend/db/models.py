import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    arxiv_id: Mapped[str] = mapped_column(String, primary_key=True)
    ingestion_status: Mapped[str] = mapped_column(String, default="pending")
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)


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

    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="session", order_by="Message.created_at"
    )
    cited_papers: Mapped[list["CitedPaper"]] = relationship(
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

    session: Mapped["Session"] = relationship("Session", back_populates="messages")


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

    session: Mapped["Session"] = relationship(
        "Session", back_populates="cited_papers"
    )
