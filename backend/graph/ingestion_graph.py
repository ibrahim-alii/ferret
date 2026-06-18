from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from backend.ingestion.ingest import ingest_paper


class IngestionState(TypedDict):
    arxiv_id: str


async def _ingest_node(state: IngestionState) -> dict:
    await ingest_paper(state["arxiv_id"])
    return {}


def _build_ingestion_graph():
    builder = StateGraph(IngestionState)
    builder.add_node("ingest", _ingest_node)
    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", END)
    return builder.compile()


ingestion_graph = _build_ingestion_graph()


async def run_ingestion(arxiv_id: str) -> None:
    await ingestion_graph.ainvoke({"arxiv_id": arxiv_id})
