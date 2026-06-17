"""Session and message endpoints with SSE streaming."""

import json
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


@router.post("/{session_id}/messages")
async def post_message(
    session_id: str,
    body: PostMessageRequest,
    db: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Persist user message before starting the stream
    user_msg = Message(
        session_id=session_id,
        role="user",
        content=body.content,
    )
    db.add(user_msg)
    await db.commit()
    await db.refresh(user_msg)

    # Build chat history from prior messages (excludes the current user turn)
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at)
    )
    prior_messages = result.scalars().all()
    chat_history = [
        {"role": m.role, "content": m.content}
        for m in prior_messages
        if m.message_id != user_msg.message_id
    ]

    return StreamingResponse(
        _stream(
            session=session,
            user_msg=user_msg,
            user_content=body.content,
            chat_history=chat_history,
            db=db,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream(
    session: Session,
    user_msg: Message,
    user_content: str,
    chat_history: list[dict],
    db: AsyncSession,
) -> AsyncIterator[str]:
    assistant_content_parts: list[str] = []
    assistant_msg_id: str | None = None
    cited_arxiv_ids: list[str] = []

    try:
        async for event in astream_chat(
            mode=session.mode,
            paper_id=session.paper_id,
            user_message=user_content,
            chat_history=chat_history,
        ):
            event_type = event.get("type", "token")

            if event_type == "token":
                assistant_content_parts.append(event.get("content", ""))
                yield _sse("token", {"content": event.get("content", "")})

            elif event_type == "interim_message":
                yield _sse("interim_message", {"content": event.get("content", "")})

            elif event_type == "citation":
                arxiv_id = event.get("arxiv_id", "")
                cited_arxiv_ids.append(arxiv_id)
                cited = CitedPaper(
                    session_id=session.session_id,
                    message_id=assistant_msg_id,
                    arxiv_id=arxiv_id,
                    title=event.get("title"),
                )
                db.add(cited)
                await db.commit()
                yield _sse("citation", {"arxiv_id": arxiv_id, "title": event.get("title")})

            elif event_type == "done":
                final_content = event.get("content", "".join(assistant_content_parts))

                assistant_msg = Message(
                    session_id=session.session_id,
                    role="assistant",
                    content=final_content,
                )
                db.add(assistant_msg)
                await db.commit()
                await db.refresh(assistant_msg)
                assistant_msg_id = assistant_msg.message_id

                # Back-fill message_id on cited_papers written before assistant msg existed
                if cited_arxiv_ids:
                    from sqlalchemy import update as sa_update
                    await db.execute(
                        sa_update(CitedPaper)
                        .where(
                            CitedPaper.session_id == session.session_id,
                            CitedPaper.message_id.is_(None),
                        )
                        .values(message_id=assistant_msg_id)
                    )
                    await db.commit()

                yield _sse(
                    "done",
                    {"content": final_content, "cited_papers": cited_arxiv_ids},
                )
    except GeneratorExit:
        pass  # client disconnected; no server error


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
    messages = result.scalars().all()
    return [MessageResponse.model_validate(m) for m in messages]
