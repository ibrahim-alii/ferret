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
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.schemas import (
    MessageResponse,
    PostMessageRequest,
    PostSessionRequest,
    PostSessionResponse,
)
from backend.db.models import CitedPaper, Message, Session
from backend.db.session import get_session
from backend.graph.entrypoint import astream_chat

router = APIRouter(prefix="/sessions", tags=["sessions"])
logger = logging.getLogger(__name__)


@router.post("", status_code=201, response_model=PostSessionResponse)
async def post_session(
    body: PostSessionRequest,
    db: AsyncSession = Depends(get_session),
) -> PostSessionResponse:
    session = Session(mode=body.mode, paper_id=body.paper_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return PostSessionResponse(session_id=session.session_id)


@router.post(
    "/{session_id}/messages",
    responses={404: {"description": "Session not found"}},
    response_class=StreamingResponse,
)
async def post_message(
    session_id: str,
    body: PostMessageRequest,
    db: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    session = await db.get(Session, session_id)
    if session is None:
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
    buffered_citations: list[dict] = []  # [{arxiv_id, title}, ...]

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

            elif event_type == "interim_message":
                yield _sse("interim_message", {"content": event.get("content", "")})

            elif event_type == "citation":
                buffered_citations.append(
                    {"arxiv_id": event.get("arxiv_id", ""), "title": event.get("title")}
                )
                yield _sse(
                    "citation",
                    {"arxiv_id": event.get("arxiv_id", ""), "title": event.get("title")},
                )

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
                        )
                    )

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


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/{session_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    session_id: str,
    db: AsyncSession = Depends(get_session),
) -> list[MessageResponse]:
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at)
    )
    return [MessageResponse.model_validate(m) for m in result.scalars().all()]
