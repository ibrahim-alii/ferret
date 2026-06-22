"""Session and message endpoints with SSE streaming.

SSE over POST: clients consume the stream via fetch() + ReadableStream,
not native EventSource (which is GET-only and cannot send a body).

Session lifecycle note: FastAPI keeps generator-dependency sessions alive until
the StreamingResponse body iterator is exhausted (cleanup `finally` fires after
the last chunk, not when the route handler returns). Passing `db` into `_stream`
is therefore safe.
"""

import json
import logging
import os
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.limiter import limiter
from backend.api.schemas import (
    MessageResponse,
    PatchSessionRequest,
    PostMessageRequest,
    PostSessionRequest,
    PostSessionResponse,
    SessionSummary,
)
from backend.db.models import CitedPaper, Message, Paper, Session
from backend.db.session import get_session
from backend.graph.entrypoint import astream_chat
from backend.graph.nodes._llm import groq_complete

router = APIRouter(prefix="/sessions", tags=["sessions"])
logger = logging.getLogger(__name__)


def _session_title(first_user_message: str | None, session: Session) -> str:
    """Derive a sidebar label: stored AI title, else first-message snippet, else mode/paper."""
    if session.title:
        return session.title
    if first_user_message:
        snippet = first_user_message.strip().replace("\n", " ")
        return snippet[:60] + ("…" if len(snippet) > 60 else "")
    if session.mode == "deep_dive" and session.paper_id:
        return f"Deep Dive · {session.paper_id}"
    return "New chat"


async def _generate_session_title(user_message: str, assistant_answer: str) -> str | None:
    """Ask the cheap grading model for a short ChatGPT-style title. None on failure."""
    model = os.environ.get("GRADING_MODEL", "llama-3.1-8b-instant")
    try:
        raw = await groq_complete(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-6 word title summarizing this conversation. "
                        "Return only the title text — no quotes, no punctuation at the end, "
                        "no prefixes like 'Title:'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"User: {user_message[:500]}\n\nAssistant: {assistant_answer[:500]}",
                },
            ],
            max_tokens=20,
        )
    except Exception as exc:
        logger.warning("Session title generation failed: %s", exc)
        return None
    title = raw.strip().strip('"').strip()
    return title[:80] or None


@router.get("", response_model=list[SessionSummary])
async def list_sessions(
    db: AsyncSession = Depends(get_session),
    x_client_id: str | None = Header(default=None),
) -> list[SessionSummary]:
    """Recent sessions for the history sidebar, newest first, scoped to client."""
    if x_client_id is None:
        return []
    result = await db.execute(
        select(Session)
        .where(Session.client_id == x_client_id)
        .order_by(Session.created_at.desc())
        .limit(50)
    )
    sessions = result.scalars().all()
    if not sessions:
        return []

    session_ids = [s.session_id for s in sessions]
    msg_result = await db.execute(
        select(Message)
        .where(Message.session_id.in_(session_ids), Message.role == "user")
        .order_by(Message.created_at)
    )
    first_user: dict[str, str] = {}
    for m in msg_result.scalars().all():
        first_user.setdefault(m.session_id, m.content)

    return [
        SessionSummary(
            session_id=s.session_id,
            mode=s.mode,
            paper_id=s.paper_id,
            created_at=s.created_at,
            title=_session_title(first_user.get(s.session_id), s),
        )
        for s in sessions
    ]


@router.post("", status_code=201, response_model=PostSessionResponse)
async def post_session(
    body: PostSessionRequest,
    db: AsyncSession = Depends(get_session),
    x_client_id: str | None = Header(default=None),
) -> PostSessionResponse:
    if body.mode == "deep_dive":
        result = await db.execute(select(Paper).where(Paper.arxiv_id == body.paper_id))
        paper = result.scalar_one_or_none()
        if paper is None or paper.ingestion_status not in ("full", "abstract_only"):
            raise HTTPException(
                status_code=400, detail="Paper not ingested or not ready for a Deep Dive session"
            )

    session = Session(mode=body.mode, paper_id=body.paper_id, client_id=x_client_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return PostSessionResponse(session_id=session.session_id)


@router.patch("/{session_id}", response_model=SessionSummary)
async def rename_session(
    session_id: str,
    body: PatchSessionRequest,
    db: AsyncSession = Depends(get_session),
    x_client_id: str | None = Header(default=None),
) -> SessionSummary:
    """Rename a chat (sets a user-defined title for the history sidebar)."""
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    # Lenient ownership: legacy rows (client_id is None) stay accessible.
    if session.client_id is not None and session.client_id != x_client_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session.title = body.title
    db.add(session)
    await db.commit()
    await db.refresh(session)

    return SessionSummary(
        session_id=session.session_id,
        mode=session.mode,
        paper_id=session.paper_id,
        created_at=session.created_at,
        title=session.title,
    )


@router.post(
    "/{session_id}/messages",
    responses={404: {"description": "Session not found"}},
    response_class=StreamingResponse,
)
@limiter.limit("30/minute")
async def post_message(
    request: Request,
    session_id: str,
    body: PostMessageRequest,
    db: AsyncSession = Depends(get_session),
    x_client_id: str | None = Header(default=None),
) -> StreamingResponse:
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    # Lenient ownership: legacy rows (client_id is None) stay accessible.
    if session.client_id is not None and session.client_id != x_client_id:
        raise HTTPException(status_code=404, detail="Session not found")

    # Persist user message before streaming begins.
    user_msg = Message(
        session_id=session_id,
        role="user",
        content=body.content,
    )
    db.add(user_msg)
    await db.commit()
    await db.refresh(user_msg)

    # Chat history = all committed messages except the current user turn.
    result = await db.execute(
        select(Message)
        .where(
            Message.session_id == session_id,
            Message.message_id != user_msg.message_id,
        )
        .order_by(Message.created_at)
    )
    chat_history = [
        {"role": m.role, "content": m.content}
        for m in result.scalars().all()
    ]

    return StreamingResponse(
        _stream(
            session=session,
            user_content=body.content,
            chat_history=chat_history,
            db=db,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream(
    session: Session,
    user_content: str,
    chat_history: list[dict],
    db: AsyncSession,
) -> AsyncIterator[str]:
    """Yield SSE frames, persisting the assistant message and citations on done.

    Citations are buffered in memory and bulk-inserted after the assistant
    Message row is committed so message_id is never NULL on CitedPaper rows.
    """
    token_parts: list[str] = []
    buffered_citations: list[dict] = []  # [{arxiv_id, title, abstract_snippet}, ...]

    try:
        async for event in astream_chat(
            mode=session.mode,
            paper_id=session.paper_id,
            user_message=user_content,
            chat_history=chat_history,
        ):
            event_type = event.get("type", "token")

            if event_type == "token":
                token_parts.append(event.get("content", ""))
                yield _sse("token", {"content": event.get("content", "")})

            elif event_type == "status":
                yield _sse(
                    "status",
                    {"step": event.get("step", ""), "content": event.get("content", "")},
                )

            elif event_type == "interim_message":
                yield _sse("interim_message", {"content": event.get("content", "")})

            elif event_type == "citation":
                citation = {
                    "arxiv_id": event.get("arxiv_id", ""),
                    "title": event.get("title"),
                    "abstract_snippet": event.get("abstract_snippet"),
                }
                buffered_citations.append(citation)
                yield _sse("citation", citation)

            elif event_type == "done":
                final_content = event.get("content", "".join(token_parts))

                assistant_msg = Message(
                    session_id=session.session_id,
                    role="assistant",
                    content=final_content,
                )
                db.add(assistant_msg)
                await db.flush()  # populate message_id before inserting citations

                for citation in buffered_citations:
                    db.add(
                        CitedPaper(
                            session_id=session.session_id,
                            message_id=assistant_msg.message_id,
                            arxiv_id=citation["arxiv_id"],
                            title=citation["title"],
                            abstract_snippet=citation["abstract_snippet"],
                        )
                    )

                # First turn (no prior history) and no title yet → generate a short
                # AI summary title for the history sidebar. Best-effort: a failure
                # here must not break the answer the user already received.
                if not chat_history and not session.title:
                    title = await _generate_session_title(user_content, final_content)
                    if title:
                        session.title = title
                        db.add(session)

                await db.commit()

                yield _sse(
                    "done",
                    {
                        "content": final_content,
                        "cited_papers": [c["arxiv_id"] for c in buffered_citations],
                    },
                )

    except GeneratorExit:
        logger.info("SSE client disconnected for session %s", session.session_id)
        await db.rollback()
    except Exception:
        # Any failure inside the graph (LLM/embeddings/rate limits, Qdrant, etc.)
        # would otherwise close the stream with no output, leaving the client
        # hanging. Surface it as an error frame and roll back the open transaction.
        logger.exception("SSE stream failed for session %s", session.session_id)
        await db.rollback()
        yield _sse("error", {"message": "The assistant ran into an error. Please try again."})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/{session_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    session_id: str,
    db: AsyncSession = Depends(get_session),
    x_client_id: str | None = Header(default=None),
) -> list[MessageResponse]:
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    # Lenient ownership: legacy rows (client_id is None) stay accessible.
    if session.client_id is not None and session.client_id != x_client_id:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at)
    )
    return [MessageResponse.model_validate(m) for m in result.scalars().all()]
