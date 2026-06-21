"""Tests for recency-aware arXiv search in ask_corrective_node.

A recency-flavoured user message ("most recent / latest papers on X") should
make the arXiv API calls sort by submission date; a normal message should not.
External APIs are mocked — no real network.
"""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.graph.nodes import ask_corrective
from backend.graph.nodes.ask_corrective import _search_arxiv, ask_corrective_node


_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.00001v1</id>
    <title>Paper</title>
    <summary>Abstract.</summary>
  </entry>
</feed>"""


def _make_http_client():
    """An httpx.AsyncClient stand-in whose .get records params and returns Atom XML."""
    response = MagicMock()
    response.text = _ATOM_XML
    response.raise_for_status = MagicMock(return_value=None)
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


async def _run_node(user_message, monkeypatch):
    """Run ask_corrective_node with all externals mocked; return the http client."""
    monkeypatch.setenv("ASK_ARXIV_QUERY_COUNT", "2")
    monkeypatch.setenv("ASK_INGEST_CANDIDATES", "1")
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")

    http_client = _make_http_client()

    # groq_complete is called twice: query generation, then the relevance filter.
    # First return queries (one per line), then the relevant arxiv id.
    groq = AsyncMock(side_effect=["alpha\nbeta", "2501.00001"])

    with patch("httpx.AsyncClient", return_value=http_client), \
         patch.object(ask_corrective, "groq_complete", groq), \
         patch.object(ask_corrective, "run_ingestion", AsyncMock()), \
         patch.object(ask_corrective, "emit", MagicMock()):
        state = {"mode": "ask", "user_message": user_message, "retry_count": 0}
        await ask_corrective_node(state)

    return http_client


async def test_recency_message_sorts_arxiv_by_date(monkeypatch):
    http_client = await _run_node("what are the most recent papers on RAG?", monkeypatch)

    assert http_client.get.call_count >= 1
    sorted_calls = [
        c for c in http_client.get.call_args_list
        if c.kwargs.get("params", {}).get("sortBy") == "submittedDate"
        and c.kwargs.get("params", {}).get("sortOrder") == "descending"
    ]
    assert sorted_calls, "expected at least one arXiv GET sorted by submittedDate descending"


async def test_non_recency_message_does_not_sort(monkeypatch):
    http_client = await _run_node("how does RAG handle retrieval?", monkeypatch)

    assert http_client.get.call_count >= 1
    for c in http_client.get.call_args_list:
        assert "sortBy" not in c.kwargs.get("params", {})


async def test_search_arxiv_param_construction():
    response = MagicMock()
    response.text = _ATOM_XML
    response.raise_for_status = MagicMock(return_value=None)
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)

    await _search_arxiv(client, "transformers", sort_by_date=True)
    params = client.get.call_args.kwargs["params"]
    assert params["search_query"] == "all:transformers"
    assert params["sortBy"] == "submittedDate"
    assert params["sortOrder"] == "descending"

    client.get.reset_mock()
    await _search_arxiv(client, "transformers")
    params = client.get.call_args.kwargs["params"]
    assert params["search_query"] == "all:transformers"
    assert "sortBy" not in params
