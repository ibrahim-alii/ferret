from typing import AsyncIterator, Literal

from backend.graph.graph import build_graph

# Compile once at import time rather than on every request.
_graph = build_graph()


async def astream_chat(
    mode: Literal["deep_dive", "ask"],
    paper_id: str | None,
    user_message: str,
    chat_history: list[dict],
) -> AsyncIterator[dict]:
    initial_state = {
        "mode": mode,
        "paper_id": paper_id,
        "user_message": user_message,
        "chat_history": chat_history,
        "retrieved_chunks": [],
        "reranked_children": [],
        "parent_sections": [],
        "grade_result": None,
        "arxiv_queries": [],
        "citations": [],
        "interim_messages": [],
        "retry_count": 0,
        "stream_events": [],
    }
    prev_count = 0
    async for state_snapshot in _graph.astream(initial_state, stream_mode="values"):
        events = state_snapshot.get("stream_events", [])
        for event in events[prev_count:]:
            yield event
        prev_count = len(events)
