"""
Unit tests for the vectorstore module.
All Qdrant client calls are mocked; no real network required.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Environment stubs (set before any vectorstore import)
# ---------------------------------------------------------------------------
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-key")
os.environ.setdefault("QDRANT_COLLECTION_NAME", "test_chunks")
os.environ.setdefault("SPARSE_MODEL", "Qdrant/bm25")
os.environ.setdefault("VOYAGE_EMBED_DIM", "1024")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sparse_vector(indices: list[int], values: list[float]):
    sv = MagicMock()
    sv.indices = indices
    sv.values = values
    return sv


def _make_chunk_vector(
    chunk_id: str = "c1",
    paper_id: str = "p1",
    section_name: str = "Introduction",
    chunk_type: str = "child",
    dense: list[float] | None = None,
    text: str = "hello world",
):
    from backend.vectorstore.models import ChunkVector

    return ChunkVector(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section_name=section_name,
        chunk_type=chunk_type,
        dense_vector=dense or [0.1] * 1024,
        text=text,
    )


def _make_scored_result(chunk_id="c1", paper_id="p1", score=0.9):
    r = MagicMock()
    r.id = chunk_id
    r.score = score
    r.payload = {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "section_name": "Introduction",
        "chunk_type": "child",
    }
    return r


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestModels:
    def test_chunk_vector_fields(self):
        from backend.vectorstore.models import ChunkVector

        cv = ChunkVector(
            chunk_id="c1",
            paper_id="p1",
            section_name="Abstract",
            chunk_type="child",
            dense_vector=[0.0] * 1024,
            text="sample text",
        )
        assert cv.chunk_id == "c1"
        assert cv.paper_id == "p1"
        assert len(cv.dense_vector) == 1024

    def test_scored_chunk_fields(self):
        from backend.vectorstore.models import ScoredChunk

        sc = ScoredChunk(
            chunk_id="c2",
            paper_id="p2",
            section_name="Methods",
            chunk_type="child",
            score=0.87,
        )
        assert sc.score == 0.87
        assert not hasattr(sc, "text"), "ScoredChunk must not carry raw text"


# ---------------------------------------------------------------------------
# Sparse encoder
# ---------------------------------------------------------------------------


class TestSparseEncoder:
    def test_singleton_instance(self):
        from backend.vectorstore import sparse as sparse_mod

        enc1 = sparse_mod.get_sparse_encoder()
        enc2 = sparse_mod.get_sparse_encoder()
        assert enc1 is enc2, "must return the same instance every call"

    def test_encode_query_returns_indices_and_values(self):
        from backend.vectorstore import sparse as sparse_mod

        sv = sparse_mod.encode_query("attention mechanism")
        assert hasattr(sv, "indices")
        assert hasattr(sv, "values")
        assert len(sv.indices) == len(sv.values)
        assert len(sv.indices) > 0

    def test_encode_document_returns_indices_and_values(self):
        from backend.vectorstore import sparse as sparse_mod

        sv = sparse_mod.encode_document("transformer architecture paper")
        assert hasattr(sv, "indices")
        assert hasattr(sv, "values")
        assert len(sv.indices) > 0

    def test_query_and_document_use_same_encoder(self):
        from backend.vectorstore import sparse as sparse_mod

        enc = sparse_mod.get_sparse_encoder()
        with patch.object(enc, "query_embed", wraps=enc.query_embed) as mock_q:
            sparse_mod.encode_query("test")
            mock_q.assert_called_once()

        with patch.object(enc, "passage_embed", wraps=enc.passage_embed) as mock_p:
            sparse_mod.encode_document("test")
            mock_p.assert_called_once()


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------


class TestEnsureCollection:
    @pytest.mark.asyncio
    async def test_collection_created_with_dense_and_sparse_named_vectors(self):
        from backend.vectorstore.store import ensure_collection

        mock_client = AsyncMock()
        mock_client.collection_exists = AsyncMock(return_value=False)
        mock_client.create_collection = AsyncMock()

        await ensure_collection(mock_client)

        mock_client.create_collection.assert_called_once()
        _, kwargs = mock_client.create_collection.call_args
        vectors_config = kwargs.get("vectors_config", {})
        sparse_vectors_config = kwargs.get("sparse_vectors_config", {})

        assert "dense" in vectors_config, "dense named vector missing"
        dense_cfg = vectors_config["dense"]
        assert dense_cfg.size == 1024, "dense vector size must equal VOYAGE_EMBED_DIM"

        assert "sparse" in sparse_vectors_config, "sparse named vector missing"

    @pytest.mark.asyncio
    async def test_collection_creation_is_idempotent(self):
        from backend.vectorstore.store import ensure_collection

        mock_client = AsyncMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_client.create_collection = AsyncMock()

        await ensure_collection(mock_client)

        mock_client.create_collection.assert_not_called()


# ---------------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------------


class TestUpsertChunks:
    @pytest.mark.asyncio
    async def test_upsert_generates_sparse_from_text(self):
        from backend.vectorstore.store import _upsert_chunks

        chunk = _make_chunk_vector(text="neural network training")
        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()

        with patch("backend.vectorstore.store.encode_document") as mock_enc:
            mock_enc.return_value = _make_sparse_vector([1], [1.0])
            await _upsert_chunks(mock_client, [chunk])
            mock_enc.assert_called_once_with("neural network training")

    @pytest.mark.asyncio
    async def test_upsert_chunks_sends_correct_payload_fields(self):
        from backend.vectorstore.store import _upsert_chunks

        chunk = _make_chunk_vector(
            chunk_id="c99",
            paper_id="p42",
            section_name="Results",
            chunk_type="child",
            text="some text",
        )
        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()

        with patch("backend.vectorstore.store.encode_document") as mock_enc:
            mock_enc.return_value = _make_sparse_vector([0], [1.0])
            await _upsert_chunks(mock_client, [chunk])

        call_args = mock_client.upsert.call_args
        points = call_args.kwargs.get("points") or call_args.args[1]
        assert len(points) == 1
        payload = points[0].payload
        assert payload["chunk_id"] == "c99"
        assert payload["paper_id"] == "p42"
        assert payload["section_name"] == "Results"
        assert payload["chunk_type"] == "child"
        assert "text" not in payload, "raw text must NOT be stored in payload"

    @pytest.mark.asyncio
    async def test_upsert_chunks_batches_large_input(self):
        from backend.vectorstore.store import _upsert_chunks

        chunks = [_make_chunk_vector(chunk_id=str(i)) for i in range(250)]
        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()

        with patch("backend.vectorstore.store.encode_document") as mock_enc:
            mock_enc.return_value = _make_sparse_vector([0], [1.0])
            await _upsert_chunks(mock_client, chunks)

        # 250 chunks / batch_size 100 = 3 batches
        assert mock_client.upsert.call_count == 3

    @pytest.mark.asyncio
    async def test_public_upsert_chunks_uses_get_client(self):
        """Public API must not require a caller-supplied client (PLAN.md contract)."""
        from backend.vectorstore.store import upsert_chunks

        chunk = _make_chunk_vector(text="public api test")
        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=mock_client)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("backend.vectorstore.store.get_client", return_value=cm):
            with patch("backend.vectorstore.store.encode_document") as mock_enc:
                mock_enc.return_value = _make_sparse_vector([0], [1.0])
                await upsert_chunks([chunk])

        mock_client.upsert.assert_called_once()


# ---------------------------------------------------------------------------
# hybrid_search
# ---------------------------------------------------------------------------


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_hybrid_search_generates_sparse_from_query_text(self):
        from backend.vectorstore.store import _hybrid_search

        mock_client = AsyncMock()
        mock_client.query_points = AsyncMock(return_value=MagicMock(points=[]))

        with patch("backend.vectorstore.store.encode_query") as mock_enc:
            mock_enc.return_value = _make_sparse_vector([1, 2], [0.6, 0.4])
            await _hybrid_search(mock_client, [0.1] * 1024, "deep learning", None, 5)
            mock_enc.assert_called_once_with("deep learning")

    @pytest.mark.asyncio
    async def test_hybrid_search_builds_prefetch_rrf_query(self):
        from backend.vectorstore.store import _hybrid_search
        from qdrant_client.models import FusionQuery, Fusion

        mock_client = AsyncMock()
        mock_client.query_points = AsyncMock(return_value=MagicMock(points=[]))

        with patch("backend.vectorstore.store.encode_query") as mock_enc:
            mock_enc.return_value = _make_sparse_vector([1], [1.0])
            await _hybrid_search(mock_client, [0.1] * 1024, "graph neural network", None, 10)

        mock_client.query_points.assert_called_once()
        call_kwargs = mock_client.query_points.call_args.kwargs

        prefetch = call_kwargs.get("prefetch")
        assert prefetch is not None, "prefetch must be provided"
        assert len(prefetch) >= 2, "prefetch must include both dense and sparse"

        query = call_kwargs.get("query")
        assert isinstance(query, FusionQuery), "query must be FusionQuery"
        assert query.fusion == Fusion.RRF

    @pytest.mark.asyncio
    async def test_hybrid_search_with_paper_id_filter_applies_filter(self):
        from backend.vectorstore.store import _hybrid_search

        mock_client = AsyncMock()
        mock_client.query_points = AsyncMock(return_value=MagicMock(points=[]))

        with patch("backend.vectorstore.store.encode_query") as mock_enc:
            mock_enc.return_value = _make_sparse_vector([0], [1.0])
            await _hybrid_search(mock_client, [0.1] * 1024, "attention", "paper_xyz", 5)

        call_kwargs = mock_client.query_points.call_args.kwargs
        qfilter = call_kwargs.get("query_filter")
        assert qfilter is not None, "filter must be set when paper_id is provided"

    @pytest.mark.asyncio
    async def test_hybrid_search_without_paper_id_has_no_filter(self):
        from backend.vectorstore.store import _hybrid_search

        mock_client = AsyncMock()
        mock_client.query_points = AsyncMock(return_value=MagicMock(points=[]))

        with patch("backend.vectorstore.store.encode_query") as mock_enc:
            mock_enc.return_value = _make_sparse_vector([0], [1.0])
            await _hybrid_search(mock_client, [0.1] * 1024, "attention", None, 5)

        call_kwargs = mock_client.query_points.call_args.kwargs
        qfilter = call_kwargs.get("query_filter")
        assert qfilter is None, "no filter must be set in Ask mode (paper_id=None)"

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_top_n(self):
        from backend.vectorstore.store import _hybrid_search

        results = [_make_scored_result(chunk_id=str(i), score=1.0 - i * 0.1) for i in range(5)]
        mock_client = AsyncMock()
        mock_client.query_points = AsyncMock(return_value=MagicMock(points=results))

        with patch("backend.vectorstore.store.encode_query") as mock_enc:
            mock_enc.return_value = _make_sparse_vector([0], [1.0])
            out = await _hybrid_search(mock_client, [0.1] * 1024, "query", None, 5)

        assert len(out) == 5
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs.get("limit") == 5

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_empty_list_when_no_matches(self):
        from backend.vectorstore.store import _hybrid_search

        mock_client = AsyncMock()
        mock_client.query_points = AsyncMock(return_value=MagicMock(points=[]))

        with patch("backend.vectorstore.store.encode_query") as mock_enc:
            mock_enc.return_value = _make_sparse_vector([0], [1.0])
            out = await _hybrid_search(mock_client, [0.1] * 1024, "obscure query", None, 10)

        assert out == []

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_scored_chunks(self):
        from backend.vectorstore.store import _hybrid_search
        from backend.vectorstore.models import ScoredChunk

        results = [_make_scored_result("c1", "p1", 0.95)]
        mock_client = AsyncMock()
        mock_client.query_points = AsyncMock(return_value=MagicMock(points=results))

        with patch("backend.vectorstore.store.encode_query") as mock_enc:
            mock_enc.return_value = _make_sparse_vector([0], [1.0])
            out = await _hybrid_search(mock_client, [0.1] * 1024, "query", None, 5)

        assert len(out) == 1
        assert isinstance(out[0], ScoredChunk)
        assert out[0].chunk_id == "c1"
        assert out[0].paper_id == "p1"
        assert out[0].score == 0.95

    @pytest.mark.asyncio
    async def test_public_hybrid_search_uses_get_client(self):
        """Public API must not require a caller-supplied client (PLAN.md contract)."""
        from backend.vectorstore.store import hybrid_search
        from backend.vectorstore.models import ScoredChunk

        mock_client = AsyncMock()
        mock_client.query_points = AsyncMock(
            return_value=MagicMock(points=[_make_scored_result("c1", "p1", 0.8)])
        )

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=mock_client)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("backend.vectorstore.store.get_client", return_value=cm):
            with patch("backend.vectorstore.store.encode_query") as mock_enc:
                mock_enc.return_value = _make_sparse_vector([0], [1.0])
                out = await hybrid_search([0.1] * 1024, "test query", None, 5)

        assert len(out) == 1
        assert isinstance(out[0], ScoredChunk)


# ---------------------------------------------------------------------------
# Integration tests (skipped by default via pytest.ini addopts)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    @pytest.mark.asyncio
    async def test_real_qdrant_create_collection_and_upsert_and_query(self):
        from backend.vectorstore.client import get_client
        from backend.vectorstore.store import ensure_collection, upsert_chunks, hybrid_search

        async with get_client() as client:
            await ensure_collection(client)
        chunks = [_make_chunk_vector(chunk_id="integ-1", text="large language models")]
        await upsert_chunks(chunks)
        results = await hybrid_search([0.1] * 1024, "language models", None, 5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_real_qdrant_hybrid_search_paper_id_filter(self):
        from backend.vectorstore.client import get_client
        from backend.vectorstore.store import ensure_collection, upsert_chunks, hybrid_search

        async with get_client() as client:
            await ensure_collection(client)
        chunks = [_make_chunk_vector(chunk_id="integ-2", paper_id="paper-A", text="attention")]
        await upsert_chunks(chunks)
        results = await hybrid_search([0.1] * 1024, "attention", "paper-A", 5)
        for r in results:
            assert r.paper_id == "paper-A"

    @pytest.mark.asyncio
    async def test_real_qdrant_server_side_rrf_returns_fused_ranking(self):
        from backend.vectorstore.client import get_client
        from backend.vectorstore.store import ensure_collection, hybrid_search

        async with get_client() as client:
            await ensure_collection(client)
        results = await hybrid_search([0.1] * 1024, "transformer", None, 3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
