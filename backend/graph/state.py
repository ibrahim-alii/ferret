from typing import Any, Literal

from typing_extensions import TypedDict


class GraphState(TypedDict):
    mode: Literal["deep_dive", "ask"]
    paper_id: str | None
    user_message: str
    chat_history: list[dict[str, Any]]
    intent: Literal["chat", "research"] | None
    query_vector: list[float] | None
    retrieved_chunks: list[dict[str, Any]]
    reranked_children: list[dict[str, Any]]
    parent_sections: list[dict[str, Any]]
    grade_result: dict[str, Any] | None
    arxiv_queries: list[str]
    citations: list[dict[str, Any]]
    interim_messages: list[str]
    retry_count: int
