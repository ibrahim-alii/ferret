"""Unit tests for the Ask-mode MMR de-duplication node."""
import pytest

from backend.graph.nodes.mmr import mmr_node


def _chunk(cid, score, vec, **extra):
    return {"chunk_id": cid, "score": score, "dense_vector": vec, "text": str(cid), **extra}


@pytest.mark.asyncio
async def test_mmr_drops_near_duplicate(monkeypatch):
    monkeypatch.setenv("RERANK_TOP_K", "2")
    monkeypatch.setenv("MMR_LAMBDA", "0.5")
    chunks = [
        _chunk("A", 0.90, [1.0, 0.0]),
        _chunk("B", 0.85, [1.0, 0.0]),  # duplicate of A
        _chunk("C", 0.80, [0.0, 1.0]),  # diverse
    ]
    result = await mmr_node({"mode": "ask", "query_vector": [1.0, 1.0], "reranked_children": chunks})
    ids = [c["chunk_id"] for c in result["reranked_children"]]
    assert ids == ["A", "C"]  # B (duplicate) dropped in favour of diverse C


@pytest.mark.asyncio
async def test_mmr_lambda_one_is_pure_relevance_order(monkeypatch):
    monkeypatch.setenv("RERANK_TOP_K", "2")
    monkeypatch.setenv("MMR_LAMBDA", "1.0")
    chunks = [
        _chunk("A", 0.90, [1.0, 0.0]),
        _chunk("B", 0.85, [1.0, 0.0]),
        _chunk("C", 0.80, [0.0, 1.0]),
    ]
    result = await mmr_node({"mode": "ask", "query_vector": [1.0, 1.0], "reranked_children": chunks})
    ids = [c["chunk_id"] for c in result["reranked_children"]]
    assert ids == ["A", "B"]  # diversity ignored -> top scores


@pytest.mark.asyncio
async def test_mmr_noop_for_deep_dive(monkeypatch):
    monkeypatch.setenv("RERANK_TOP_K", "2")
    chunks = [_chunk(i, 0.9 - i / 10, [1.0, 0.0]) for i in range(4)]
    result = await mmr_node({"mode": "deep_dive", "query_vector": [1.0], "reranked_children": chunks})
    ids = [c["chunk_id"] for c in result["reranked_children"]]
    assert ids == [0, 1]  # untouched order, trimmed to top_k


@pytest.mark.asyncio
async def test_mmr_falls_back_when_vectors_missing(monkeypatch):
    monkeypatch.setenv("RERANK_TOP_K", "2")
    chunks = [
        {"chunk_id": "A", "score": 0.9, "text": "a"},  # no dense_vector
        {"chunk_id": "B", "score": 0.8, "text": "b"},
    ]
    result = await mmr_node({"mode": "ask", "query_vector": [1.0], "reranked_children": chunks})
    ids = [c["chunk_id"] for c in result["reranked_children"]]
    assert ids == ["A", "B"]  # graceful fallback to rerank order
