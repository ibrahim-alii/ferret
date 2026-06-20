"""Live event emission for graph nodes.

Nodes call ``emit`` to push status/token/citation events to the client the moment
they happen, via LangGraph's custom stream writer (the graph is run with
``stream_mode="custom"`` in :mod:`backend.graph.entrypoint`). Outside a graph run
(e.g. a unit test calling a node directly) there is no writer in context, so this
is a safe no-op.
"""
from typing import Any

from langgraph.config import get_stream_writer


def emit(event: dict[str, Any]) -> None:
    try:
        writer = get_stream_writer()
    except (RuntimeError, LookupError):
        return
    writer(event)
