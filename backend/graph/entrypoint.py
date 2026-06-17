"""Module 3 stub — LangGraph chat entrypoint."""

from collections.abc import AsyncIterator
from typing import Any


async def astream_chat(
    mode: str,
    paper_id: str | None,
    user_message: str,
    chat_history: list[dict[str, str]],
) -> AsyncIterator[dict[str, Any]]:
    """Stream chat events from the LangGraph graph.

    Yields dicts with a 'type' key; possible types:
      token          — {type, content}
      interim_message — {type, content}
      citation       — {type, arxiv_id, title}
      done           — {type, content, cited_papers}
    """
    yield {"type": "token", "content": "(stub) "}
    yield {"type": "done", "content": "(stub) answer", "cited_papers": []}
