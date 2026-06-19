"""
Unit tests for Module 1: Ingestion pipeline.
All external APIs (arxiv, Voyage, Qdrant) are mocked.
Integration tests are marked @pytest.mark.integration and skipped by default.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Sample fixtures
# ---------------------------------------------------------------------------

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Test Paper Title</title>
    <summary>This is the abstract text.</summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <published>2023-01-01T00:00:00Z</published>
  </entry>
</feed>"""

SAMPLE_HTML = """<html><body>
<section id="S1">
  <h2>Introduction</h2>
  <p>This paper introduces something cool.</p>
</section>
<section id="S2">
  <h2>Methods</h2>
  <p>We used method A and method B.</p>
</section>
<section id="S3">
  <h2>Results</h2>
  <p>Results were excellent.</p>
</section>
<section id="S4">
  <h2>References</h2>
  <p>[1] Smith et al. 2020</p>
</section>
<section id="S5">
  <h2>Acknowledgements</h2>
  <p>We thank funding agency X.</p>
</section>
</body></html>"""


# ---------------------------------------------------------------------------
# arxiv_client tests
# ---------------------------------------------------------------------------

class TestFetchMetadata:
    @pytest.mark.asyncio
    async def test_fetch_metadata_returns_title_abstract_authors(self):
        from backend.ingestion.arxiv_client import fetch_metadata

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_ATOM
        mock_response.raise_for_status = MagicMock()

        with patch("backend.ingestion.arxiv_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await fetch_metadata("2301.00001")

        assert result["title"] == "Test Paper Title"
        assert result["abstract"] == "This is the abstract text."
        assert "Alice Smith" in result["authors"]
        assert "Bob Jones" in result["authors"]
        assert result["published_date"] == "2023-01-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_fetch_metadata_handles_invalid_arxiv_id(self):
        from backend.ingestion.arxiv_client import fetch_metadata, ArxivNotFoundError

        empty_atom = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = empty_atom
        mock_response.raise_for_status = MagicMock()

        with patch("backend.ingestion.arxiv_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ArxivNotFoundError):
                await fetch_metadata("9999.99999")

    @pytest.mark.asyncio
    async def test_arxiv_prefer_html_false_skips_html_fetch(self):
        """When ARXIV_PREFER_HTML=false, fetch_html is never called."""
        from backend.ingestion import ingest

        fetch_html_mock = AsyncMock(return_value="<html/>")
        fetch_pdf_mock = AsyncMock(return_value=b"")

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.dict("os.environ", {"ARXIV_PREFER_HTML": "false"}),
            patch("backend.ingestion.ingest.fetch_metadata", AsyncMock(return_value={
                "title": "T", "abstract": "A", "authors": ["X"],
                "published_date": "2023-01-01T00:00:00Z",
            })),
            patch("backend.ingestion.ingest.fetch_html", fetch_html_mock),
            patch("backend.ingestion.ingest.download_pdf", fetch_pdf_mock),
            patch("backend.ingestion.ingest.parse", return_value=[]),
            patch("backend.ingestion.ingest.filter_sections", return_value=[]),
            patch("backend.ingestion.ingest.chunk_sections", return_value=([], [])),
            patch("backend.ingestion.ingest.embed_chunks", AsyncMock(return_value=[])),
            patch("backend.ingestion.ingest.upsert_chunks", AsyncMock()),
            patch("backend.ingestion.ingest.async_session", return_value=mock_ctx),
        ):
            await ingest.ingest_paper("2301.00001")

        fetch_html_mock.assert_not_called()
        fetch_pdf_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_and_metadata_fetch_run_concurrently(self):
        """asyncio.gather is used so both calls start before either finishes."""
        from backend.ingestion import ingest

        call_order: list[str] = []

        async def fake_fetch_metadata(arxiv_id):
            call_order.append("metadata_start")
            await asyncio.sleep(0.01)
            call_order.append("metadata_end")
            return {
                "title": "T", "abstract": "A", "authors": ["X"],
                "published_date": "2023-01-01T00:00:00Z",
            }

        async def fake_fetch_html(arxiv_id):
            call_order.append("html_start")
            await asyncio.sleep(0.01)
            call_order.append("html_end")
            return None

        async def fake_download_pdf(arxiv_id):
            call_order.append("pdf_start")
            await asyncio.sleep(0.01)
            call_order.append("pdf_end")
            return b""

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("backend.ingestion.ingest.fetch_metadata", side_effect=fake_fetch_metadata),
            patch("backend.ingestion.ingest.fetch_html", side_effect=fake_fetch_html),
            patch("backend.ingestion.ingest.download_pdf", side_effect=fake_download_pdf),
            patch("backend.ingestion.ingest.parse", return_value=[]),
            patch("backend.ingestion.ingest.filter_sections", return_value=[]),
            patch("backend.ingestion.ingest.chunk_sections", return_value=([], [])),
            patch("backend.ingestion.ingest.embed_chunks", AsyncMock(return_value=[])),
            patch("backend.ingestion.ingest.upsert_chunks", AsyncMock()),
            patch("backend.ingestion.ingest.async_session", return_value=mock_session_ctx),
        ):
            await ingest.ingest_paper("2301.00001")

        # Both metadata and html/pdf must start before either finishes
        assert call_order.index("metadata_start") < call_order.index("html_end")
        assert call_order.index("html_start") < call_order.index("metadata_end")


# ---------------------------------------------------------------------------
# parser tests
# ---------------------------------------------------------------------------

class TestParser:
    def test_parse_prefers_arxiv_html_when_available(self):
        from backend.ingestion.parser import parse

        sections = parse(html_content=SAMPLE_HTML, pdf_bytes=None)

        section_names = [s[0].lower() for s in sections]
        assert any("introduction" in n for n in section_names)
        assert any("methods" in n for n in section_names)

    def test_parse_falls_back_to_pymupdf_when_html_missing(self):
        from backend.ingestion.parser import parse
        import fitz  # PyMuPDF

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Introduction\nThis is the intro.", fontsize=14)
        pdf_bytes = doc.tobytes()
        doc.close()

        sections = parse(html_content=None, pdf_bytes=pdf_bytes)

        assert isinstance(sections, list)
        for item in sections:
            assert isinstance(item, tuple) and len(item) == 2
            assert isinstance(item[0], str) and isinstance(item[1], str)

    def test_html_and_pdf_paths_return_same_section_tuple_shape(self):
        from backend.ingestion.parser import parse

        html_sections = parse(html_content=SAMPLE_HTML, pdf_bytes=None)
        for item in html_sections:
            assert isinstance(item, tuple) and len(item) == 2
            assert all(isinstance(x, str) for x in item)

    def test_parse_failure_falls_back_to_abstract_only(self):
        from backend.ingestion.parser import parse

        sections = parse(html_content=None, pdf_bytes=None)
        assert sections == []


# ---------------------------------------------------------------------------
# filter tests
# ---------------------------------------------------------------------------

class TestFilter:
    def _make_sections(self, names):
        return [(name, f"Content of {name}") for name in names]

    def test_section_filter_drops_references_section(self):
        from backend.ingestion.filter import filter_sections

        sections = self._make_sections(["Introduction", "Methods", "References"])
        result = filter_sections(sections)
        names = [s[0] for s in result]
        assert "References" not in names

    def test_section_filter_drops_acknowledgements_section(self):
        from backend.ingestion.filter import filter_sections

        sections = self._make_sections(["Introduction", "Acknowledgements", "Results"])
        result = filter_sections(sections)
        names = [s[0] for s in result]
        assert "Acknowledgements" not in names

    def test_section_filter_keeps_methods_and_results_sections(self):
        from backend.ingestion.filter import filter_sections

        sections = self._make_sections(["Methods", "Results", "References"])
        result = filter_sections(sections)
        names = [s[0] for s in result]
        assert "Methods" in names
        assert "Results" in names

    def test_section_filter_keeps_abstract_always(self):
        from backend.ingestion.filter import filter_sections

        sections = self._make_sections(["Abstract", "References", "Acknowledgements"])
        result = filter_sections(sections)
        names = [s[0] for s in result]
        assert "Abstract" in names

    def test_section_filter_list_is_configurable(self):
        from backend.ingestion.filter import filter_sections

        sections = self._make_sections(["Introduction", "Methods", "CustomNoise"])
        result = filter_sections(sections, drop_list=["CustomNoise"])
        names = [s[0] for s in result]
        assert "Introduction" in names
        assert "Methods" in names
        assert "CustomNoise" not in names

    def test_section_filter_drops_equation_only_sections(self):
        """Sections with almost no prose after stripping LaTeX are dropped."""
        from backend.ingestion.filter import filter_sections

        equation_section = ("Proof", "$x^2 + y^2 = z^2$ \\sum_{i} x_i")
        prose_section = ("Methods", "We ran experiments on three datasets. " * 10)
        result = filter_sections(
            [equation_section, prose_section],
            equation_min_prose_chars=100,
        )
        names = [s[0] for s in result]
        assert "Proof" not in names
        assert "Methods" in names


# ---------------------------------------------------------------------------
# chunker tests
# ---------------------------------------------------------------------------

class TestChunker:
    def test_chunking_creates_one_parent_per_section_plus_abstract_parent(self):
        from backend.ingestion.chunker import chunk_sections

        sections = [
            ("Introduction", "Intro text " * 20),
            ("Methods", "Methods text " * 20),
        ]
        parents, children = chunk_sections(
            paper_id=1, abstract="The abstract.", sections=sections
        )
        parent_section_names = {p.section_name for p in parents}
        assert "Abstract" in parent_section_names
        assert "Introduction" in parent_section_names
        assert "Methods" in parent_section_names
        assert len(parents) == 3

    def test_chunking_creates_child_chunks_linked_to_parent_via_parent_chunk_id(self):
        """
        parent_chunk_id is an FK wired after DB flush in ingest.py.
        The chunker records the link via _parent_ref so ingest.py can set the FK.
        """
        from backend.ingestion.chunker import chunk_sections

        sections = [("Introduction", "Word " * 600)]
        parents, children = chunk_sections(
            paper_id=1, abstract="Short abstract.", sections=sections
        )
        assert len(children) > 0
        for child in children:
            assert hasattr(child, "_parent_ref") and child._parent_ref is not None
            assert child.chunk_type == "child"

    def test_child_chunks_respect_token_window_size(self):
        from backend.ingestion.chunker import chunk_sections
        import tiktoken
        import os

        token_limit = int(os.environ.get("CHILD_CHUNK_TOKENS", "512"))
        enc = tiktoken.get_encoding("cl100k_base")

        sections = [("Methods", "token " * 2000)]
        _, children = chunk_sections(
            paper_id=1, abstract="Abstract.", sections=sections
        )
        for child in children:
            assert len(enc.encode(child.text)) <= token_limit + 10


# ---------------------------------------------------------------------------
# embedder tests
# ---------------------------------------------------------------------------

class TestEmbedder:
    def _make_mock_chunk(self, text="hello world", chunk_id=1):
        chunk = MagicMock()
        chunk.id = chunk_id
        chunk.text = text
        return chunk

    @pytest.mark.asyncio
    async def test_embedding_batches_chunks_not_one_call_per_chunk(self):
        from backend.ingestion.embedder import embed_chunks

        chunks = [self._make_mock_chunk(f"text {i}", i) for i in range(10)]
        mock_embed_result = MagicMock()
        mock_embed_result.embeddings = [[0.1] * 1024 for _ in range(10)]

        with patch("backend.ingestion.embedder.voyageai.AsyncClient") as mock_cls:
            mock_voyage = AsyncMock()
            mock_voyage.embed = AsyncMock(return_value=mock_embed_result)
            mock_cls.return_value = mock_voyage

            with patch.dict("os.environ", {"VOYAGE_BATCH_SIZE": "10", "VOYAGE_API_KEY": "test-key"}):
                result = await embed_chunks(chunks)

        assert mock_voyage.embed.call_count == 1
        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_embedding_respects_bounded_concurrency(self):
        from backend.ingestion.embedder import embed_chunks

        max_concurrency = 2
        active: list[int] = []
        max_active = [0]

        async def fake_embed(texts, model, input_type):
            active.append(1)
            max_active[0] = max(max_active[0], len(active))
            await asyncio.sleep(0.01)
            active.pop()
            result = MagicMock()
            result.embeddings = [[0.1] * 1024 for _ in texts]
            return result

        chunks = [self._make_mock_chunk(f"text {i}", i) for i in range(10)]

        with patch("backend.ingestion.embedder.voyageai.AsyncClient") as mock_cls:
            mock_voyage = AsyncMock()
            mock_voyage.embed = fake_embed
            mock_cls.return_value = mock_voyage

            with patch.dict("os.environ", {
                "VOYAGE_BATCH_SIZE": "2",
                "VOYAGE_MAX_CONCURRENCY": str(max_concurrency),
                "VOYAGE_API_KEY": "test-key",
            }):
                await embed_chunks(chunks)

        assert max_active[0] <= max_concurrency


# ---------------------------------------------------------------------------
# ingest orchestration tests
# ---------------------------------------------------------------------------

class TestIngest:
    def _make_paper_mock(self, status="full", arxiv_id="2301.00001"):
        paper = MagicMock()
        paper.id = 1
        paper.arxiv_id = arxiv_id
        paper.title = "Test Paper"
        paper.abstract = "Test abstract."
        paper.ingestion_status = status
        return paper

    def _make_session_ctx(self, existing_paper=None):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=existing_paper)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        return mock_ctx, mock_session

    @pytest.mark.asyncio
    async def test_only_child_chunks_embedded_and_upserted(self):
        from backend.ingestion import ingest
        from backend.ingestion.chunker import Chunk as ChunkModel

        mock_ctx, _ = self._make_session_ctx(existing_paper=None)
        upsert_mock = AsyncMock()

        parent = MagicMock(spec=ChunkModel)
        parent.id = 10
        parent.chunk_type = "parent"
        parent.section_name = "Introduction"
        parent.text = "intro text"
        parent.paper_id = 1

        child = MagicMock(spec=ChunkModel)
        child.id = 11
        child.chunk_type = "child"
        child.section_name = "Introduction"
        child.text = "child text"
        child.parent_chunk_id = 10
        child.paper_id = 1

        with (
            patch("backend.ingestion.ingest.fetch_metadata", AsyncMock(return_value={
                "title": "T", "abstract": "A", "authors": ["X"],
                "published_date": "2023-01-01T00:00:00Z",
            })),
            patch("backend.ingestion.ingest.fetch_html", AsyncMock(return_value=None)),
            patch("backend.ingestion.ingest.download_pdf", AsyncMock(return_value=b"")),
            patch("backend.ingestion.ingest.parse", return_value=[("Introduction", "text")]),
            patch("backend.ingestion.ingest.filter_sections", return_value=[("Introduction", "text")]),
            patch("backend.ingestion.ingest.chunk_sections", return_value=([parent], [child])),
            patch("backend.ingestion.ingest.embed_chunks", AsyncMock(return_value=[[0.1] * 1024])),
            patch("backend.ingestion.ingest.upsert_chunks", upsert_mock),
            patch("backend.ingestion.ingest.async_session", return_value=mock_ctx),
        ):
            await ingest.ingest_paper("2301.00001")

        upsert_mock.assert_called_once()
        upserted = upsert_mock.call_args[0][0]
        assert len(upserted) == 1
        assert upserted[0].chunk_type == "child"

    @pytest.mark.asyncio
    async def test_upsert_called_with_correct_chunkvector_fields(self):
        from backend.ingestion import ingest
        from backend.ingestion.chunker import Chunk as ChunkModel

        mock_ctx, _ = self._make_session_ctx(existing_paper=None)
        upsert_mock = AsyncMock()

        child = MagicMock(spec=ChunkModel)
        child.id = 11
        child.chunk_type = "child"
        child.section_name = "Methods"
        child.text = "method text"
        child.parent_chunk_id = 10
        child.paper_id = 1

        with (
            patch("backend.ingestion.ingest.fetch_metadata", AsyncMock(return_value={
                "title": "T", "abstract": "A", "authors": ["X"],
                "published_date": "2023-01-01T00:00:00Z",
            })),
            patch("backend.ingestion.ingest.fetch_html", AsyncMock(return_value=None)),
            patch("backend.ingestion.ingest.download_pdf", AsyncMock(return_value=b"")),
            patch("backend.ingestion.ingest.parse", return_value=[("Methods", "text")]),
            patch("backend.ingestion.ingest.filter_sections", return_value=[("Methods", "text")]),
            patch("backend.ingestion.ingest.chunk_sections", return_value=([], [child])),
            patch("backend.ingestion.ingest.embed_chunks", AsyncMock(return_value=[[0.5] * 1024])),
            patch("backend.ingestion.ingest.upsert_chunks", upsert_mock),
            patch("backend.ingestion.ingest.async_session", return_value=mock_ctx),
        ):
            await ingest.ingest_paper("2301.00001")

        upsert_mock.assert_called_once()
        cv = upsert_mock.call_args[0][0][0]
        assert hasattr(cv, "dense_vector")
        assert hasattr(cv, "text")
        assert not hasattr(cv, "sparse_vector")
        assert cv.chunk_type == "child"
        assert cv.section_name == "Methods"

    @pytest.mark.asyncio
    async def test_ingest_paper_marks_status_full_on_success(self):
        from backend.ingestion import ingest
        from backend.ingestion.chunker import Chunk as ChunkModel

        mock_ctx, _ = self._make_session_ctx(existing_paper=None)

        child = MagicMock(spec=ChunkModel)
        child.id = 11
        child.chunk_type = "child"
        child.section_name = "Introduction"
        child.text = "intro text"
        child.parent_chunk_id = 10
        child.paper_id = 1

        with (
            patch("backend.ingestion.ingest.fetch_metadata", AsyncMock(return_value={
                "title": "T", "abstract": "A", "authors": ["X"],
                "published_date": "2023-01-01T00:00:00Z",
            })),
            patch("backend.ingestion.ingest.fetch_html", AsyncMock(return_value=None)),
            patch("backend.ingestion.ingest.download_pdf", AsyncMock(return_value=b"")),
            patch("backend.ingestion.ingest.parse", return_value=[("Introduction", "text")]),
            patch("backend.ingestion.ingest.filter_sections", return_value=[("Introduction", "text")]),
            patch("backend.ingestion.ingest.chunk_sections", return_value=([], [child])),
            patch("backend.ingestion.ingest.embed_chunks", AsyncMock(return_value=[[0.5] * 1024])),
            patch("backend.ingestion.ingest.upsert_chunks", AsyncMock()),
            patch("backend.ingestion.ingest.async_session", return_value=mock_ctx),
        ):
            result = await ingest.ingest_paper("2301.00001")

        assert result.ingestion_status == "full"

    @pytest.mark.asyncio
    async def test_ingest_paper_is_idempotent_for_already_full_paper(self):
        from backend.ingestion import ingest

        existing = self._make_paper_mock(status="full")
        mock_ctx, _ = self._make_session_ctx(existing_paper=existing)
        fetch_meta_mock = AsyncMock()

        with (
            patch("backend.ingestion.ingest.fetch_metadata", fetch_meta_mock),
            patch("backend.ingestion.ingest.async_session", return_value=mock_ctx),
        ):
            result = await ingest.ingest_paper("2301.00001")

        fetch_meta_mock.assert_not_called()
        assert result.ingestion_status == "full"

    @pytest.mark.asyncio
    async def test_ingest_paper_writes_chunks_to_sqlite_with_correct_paper_id(self):
        from backend.ingestion import ingest
        from backend.ingestion.chunker import Chunk as ChunkModel

        mock_ctx, mock_session = self._make_session_ctx(existing_paper=None)
        added_objects: list = []
        mock_session.add.side_effect = lambda obj: added_objects.append(obj)

        parent = MagicMock(spec=ChunkModel)
        parent.id = None
        parent.paper_id = 1
        parent.chunk_type = "parent"
        parent.section_name = "Intro"
        parent.text = "intro"
        parent.parent_chunk_id = None
        parent.token_count = 5

        child = MagicMock(spec=ChunkModel)
        child.id = None
        child.paper_id = 1
        child.chunk_type = "child"
        child.section_name = "Intro"
        child.text = "child"
        child.parent_chunk_id = None
        child.token_count = 3

        with (
            patch("backend.ingestion.ingest.fetch_metadata", AsyncMock(return_value={
                "title": "T", "abstract": "A", "authors": ["X"],
                "published_date": "2023-01-01T00:00:00Z",
            })),
            patch("backend.ingestion.ingest.fetch_html", AsyncMock(return_value=None)),
            patch("backend.ingestion.ingest.download_pdf", AsyncMock(return_value=b"")),
            patch("backend.ingestion.ingest.parse", return_value=[("Intro", "text")]),
            patch("backend.ingestion.ingest.filter_sections", return_value=[("Intro", "text")]),
            patch("backend.ingestion.ingest.chunk_sections", return_value=([parent], [child])),
            patch("backend.ingestion.ingest.embed_chunks", AsyncMock(return_value=[[0.1] * 1024])),
            patch("backend.ingestion.ingest.upsert_chunks", AsyncMock()),
            patch("backend.ingestion.ingest.async_session", return_value=mock_ctx),
        ):
            await ingest.ingest_paper("2301.00001")

        chunk_objects = [o for o in added_objects if hasattr(o, "chunk_type")]
        assert len(chunk_objects) >= 2


# ---------------------------------------------------------------------------
# Integration tests (skipped by default via pyproject.toml addopts)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    @pytest.mark.asyncio
    async def test_real_arxiv_fetch_known_paper_id(self):
        from backend.ingestion.arxiv_client import fetch_metadata
        result = await fetch_metadata("1706.03762")
        assert "attention" in result["title"].lower()

    @pytest.mark.asyncio
    async def test_real_voyage_embedding_call(self):
        from backend.ingestion.embedder import embed_chunks
        chunk = MagicMock()
        chunk.text = "Transformers use self-attention mechanisms."
        result = await embed_chunks([chunk])
        assert len(result) == 1
        assert len(result[0]) > 0

    @pytest.mark.asyncio
    async def test_full_ingest_pipeline_against_real_apis_small_paper(self):
        from backend.ingestion.ingest import ingest_paper
        record = await ingest_paper("1706.03762")
        assert record.ingestion_status in ("full", "abstract_only")
        assert record.title
        assert record.abstract
