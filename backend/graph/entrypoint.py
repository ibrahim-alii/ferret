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
    }
    # stream_mode="custom" surfaces each event a node passes to its stream writer
    # the instant it is produced, so status/token/citation events flush live
    # rather than being batched after each node returns.
    async for event in _graph.astream(initial_state, stream_mode="custom"):
        yield event
