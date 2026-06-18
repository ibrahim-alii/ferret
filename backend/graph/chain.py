"""LangGraph chat chain — stub until Module 3 is integrated."""
from __future__ import annotations

from typing import AsyncIterator, Optional


async def astream_chat(
    mode: str,
    paper_id: Optional[str],
    user_message: str,
    chat_history: list,
) -> AsyncIterator[dict]:
    """Stream chat events from the graph."""
    return
    yield  # make this an async generator
