"""
TDD tests for the CRAG graph (Module 3).
All tests use pytest-mock to avoid hitting real external APIs.
asyncio_mode = "auto" is set in pyproject.toml so async tests run automatically.
"""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(chunk_id="c1", parent_chunk_id="p1", paper_id="arxiv123", text="text", score=0.8):
    return {
        "chunk_id": chunk_id,
        "parent_chunk_id": parent_chunk_id,
        "paper_id": paper_id,
        "text": text,
        "score": score,
    }


def make_state(**overrides):
    base = {
        "mode": "ask",
        "paper_id": None,
        "user_message": "What is CRAG?",
        "chat_history": [],
        "retrieved_chunks": [],
        "reranked_children": [],
        "parent_sections": [],
        "grade_result": None,
        "arxiv_queries": [],
        "citations": [],
        "interim_messages": [],
        "retry_count": 0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Retrieve node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_node_deep_dive_passes_paper_id_filter(monkeypatch):
    monkeypatch.setenv("RERANK_CANDIDATE_COUNT", "30")

    mock_hybrid = AsyncMock(return_value=[])

    with patch("backend.graph.nodes.retrieve.embed_query", AsyncMock(return_value=[0.1, 0.2, 0.3])), \
         patch("backend.graph.nodes.retrieve.hybrid_search", mock_hybrid):
        from backend.graph.nodes.retrieve import retrieve_node
        state = make_state(mode="deep_dive", paper_id="arxiv123")
        result = await retrieve_node(state)

    call_kwargs = mock_hybrid.call_args
    assert call_kwargs.kwargs.get("paper_id") == "arxiv123" or call_kwargs.args[2] == "arxiv123"


@pytest.mark.asyncio
async def test_retrieve_node_ask_passes_no_paper_id_filter(monkeypatch):
    monkeypatch.setenv("RERANK_CANDIDATE_COUNT", "30")

    mock_hybrid = AsyncMock(return_value=[])

    with patch("backend.graph.nodes.retrieve.embed_query", AsyncMock(return_value=[0.1, 0.2, 0.3])), \
         patch("backend.graph.nodes.retrieve.hybrid_search", mock_hybrid):
        from backend.graph.nodes.retrieve import retrieve_node
        state = make_state(mode="ask", paper_id=None)
        result = await retrieve_node(state)

    call_kwargs = mock_hybrid.call_args
    # paper_id should be None for ask mode
    paper_id_val = call_kwargs.kwargs.get("paper_id", call_kwargs.args[2] if len(call_kwargs.args) > 2 else None)
    assert paper_id_val is None


@pytest.mark.asyncio
async def test_retrieve_node_calls_hybrid_search_with_query_dense_and_text(monkeypatch):
    monkeypatch.setenv("RERANK_CANDIDATE_COUNT", "30")

    mock_hybrid = AsyncMock(return_value=[])

    with patch("backend.graph.nodes.retrieve.embed_query", AsyncMock(return_value=[0.1, 0.2, 0.3])), \
         patch("backend.graph.nodes.retrieve.hybrid_search", mock_hybrid):
        from backend.graph.nodes.retrieve import retrieve_node
        state = make_state(user_message="What is CRAG?")
        await retrieve_node(state)

    assert mock_hybrid.called
    # Check query_text is passed
    call_kwargs = mock_hybrid.call_args
    query_text_val = call_kwargs.kwargs.get("query_text", None)
    if query_text_val is None and len(call_kwargs.args) > 1:
        query_text_val = call_kwargs.args[1]
    assert query_text_val == "What is CRAG?"


# ---------------------------------------------------------------------------
# Rerank node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rerank_node_calls_jina_rerank_with_top_n_candidates(monkeypatch):
    monkeypatch.setenv("RERANK_CANDIDATE_COUNT", "30")
    monkeypatch.setenv("RERANK_POOL_K", "5")

    mock_rerank = AsyncMock(return_value={"results": [{"index": 0, "relevance_score": 0.9}]})

    chunks = [make_chunk(chunk_id=f"c{i}") for i in range(30)]

    with patch("backend.graph.nodes.rerank._rerank_with_retry", mock_rerank):
        from backend.graph.nodes.rerank import rerank_node
        await rerank_node(make_state(retrieved_chunks=chunks))

    assert mock_rerank.called
    payload = mock_rerank.call_args.args[0]
    assert len(payload["documents"]) == 30  # RERANK_CANDIDATE_COUNT
    assert payload["model"]                 # a model identifier is set
    assert payload["top_n"] == 5            # RERANK_POOL_K (mmr_node trims to RERANK_TOP_K)


@pytest.mark.asyncio
async def test_rerank_node_returns_top_k(monkeypatch):
    monkeypatch.setenv("RERANK_CANDIDATE_COUNT", "30")
    monkeypatch.setenv("RERANK_POOL_K", "5")

    # 10 Jina results sorted by score descending
    results = [{"index": i, "relevance_score": 1.0 - i * 0.05} for i in range(10)]
    mock_rerank = AsyncMock(return_value={"results": results})

    chunks = [make_chunk(chunk_id=f"c{i}", text=f"text {i}") for i in range(10)]

    with patch("backend.graph.nodes.rerank._rerank_with_retry", mock_rerank):
        from backend.graph.nodes.rerank import rerank_node
        result = await rerank_node(make_state(retrieved_chunks=chunks))

    assert len(result["reranked_children"]) == 5  # RERANK_POOL_K; mmr trims to RERANK_TOP_K


@pytest.mark.asyncio
async def test_rerank_node_drops_empty_text_chunks_before_jina(monkeypatch):
    """Jina rejects empty strings; chunks with no text (e.g. a Qdrant hit with no
    SQLite row) must be filtered out so a conversational query never aborts."""
    monkeypatch.setenv("RERANK_TOP_K", "5")

    mock_rerank = AsyncMock(return_value={"results": [{"index": 0, "relevance_score": 0.9}]})

    # Mix of empty / whitespace-only / real text.
    chunks = [
        make_chunk(chunk_id="c0", text=""),
        make_chunk(chunk_id="c1", text="   "),
        make_chunk(chunk_id="c2", text="real content"),
    ]

    with patch("backend.graph.nodes.rerank._rerank_with_retry", mock_rerank):
        from backend.graph.nodes.rerank import rerank_node
        result = await rerank_node(make_state(retrieved_chunks=chunks))

    payload = mock_rerank.call_args.args[0]
    assert payload["documents"] == ["real content"]  # no empty strings reach Jina
    assert len(result["reranked_children"]) == 1


@pytest.mark.asyncio
async def test_rerank_node_returns_empty_when_all_text_blank(monkeypatch):
    """If every retrieved chunk has empty text, skip Jina entirely (empty list is invalid)."""
    mock_rerank = AsyncMock()

    chunks = [make_chunk(chunk_id=f"c{i}", text="") for i in range(5)]

    with patch("backend.graph.nodes.rerank._rerank_with_retry", mock_rerank):
        from backend.graph.nodes.rerank import rerank_node
        result = await rerank_node(make_state(retrieved_chunks=chunks))

    assert result == {"reranked_children": []}
    mock_rerank.assert_not_called()


@pytest.mark.asyncio
async def test_rerank_with_retry_retries_transient_then_succeeds(monkeypatch):
    """A transient network error is retried; a subsequent 200 returns the parsed body."""
    monkeypatch.setenv("JINA_RETRY_BASE_WAIT", "0")  # don't actually sleep
    import httpx
    from backend.graph.nodes import rerank as rr

    ok = MagicMock()
    ok.raise_for_status = MagicMock()
    ok.json = MagicMock(return_value={"results": []})
    post = AsyncMock(side_effect=[httpx.ConnectError("boom"), ok])

    class FakeClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            client = MagicMock()
            client.post = post
            return client
        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    out = await rr._rerank_with_retry({"documents": ["x"]}, {})
    assert out == {"results": []}
    assert post.await_count == 2


# ---------------------------------------------------------------------------
# Expand node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_expand_node_fetches_parent_sections_for_reranked_children(monkeypatch):
    monkeypatch.setenv("GENERATION_MAX_PARENTS", "5")
    monkeypatch.setenv("SQLITE_DB_PATH", "test.db")

    chunks = [make_chunk(chunk_id="c1", parent_chunk_id="p1", paper_id="arxiv1", score=0.9)]

    mock_row = {"id": "p1", "parent_chunk_id": None, "paper_id": "arxiv1", "paper_title": "A Paper", "published_date": "2023-01-01T00:00:00Z", "abstract": "An abstract.", "content_type": "text", "media_url": None, "section_name": "Intro", "text": "Parent text"}

    mock_cursor = AsyncMock()
    mock_cursor.fetchone = AsyncMock(return_value=mock_row)
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with patch("aiosqlite.connect", return_value=mock_conn):
        from backend.graph.nodes.expand import expand_node
        state = make_state(reranked_children=chunks)
        result = await expand_node(state)

    assert len(result["parent_sections"]) == 1
    assert result["parent_sections"][0]["text"] == "Parent text"
    assert result["parent_sections"][0]["published_date"] == "2023-01-01T00:00:00Z"


@pytest.mark.asyncio
async def test_expand_node_dedupes_shared_parents(monkeypatch):
    monkeypatch.setenv("GENERATION_MAX_PARENTS", "5")
    monkeypatch.setenv("SQLITE_DB_PATH", "test.db")

    # Two children sharing the same parent
    chunks = [
        make_chunk(chunk_id="c1", parent_chunk_id="p1", paper_id="arxiv1", score=0.9),
        make_chunk(chunk_id="c2", parent_chunk_id="p1", paper_id="arxiv1", score=0.8),
    ]

    mock_row = {"id": "p1", "parent_chunk_id": None, "paper_id": "arxiv1", "paper_title": "A Paper", "published_date": "2023-01-01T00:00:00Z", "abstract": "An abstract.", "content_type": "text", "media_url": None, "section_name": "Intro", "text": "Parent text"}

    mock_cursor = AsyncMock()
    mock_cursor.fetchone = AsyncMock(return_value=mock_row)
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with patch("aiosqlite.connect", return_value=mock_conn):
        from backend.graph.nodes.expand import expand_node
        state = make_state(reranked_children=chunks)
        result = await expand_node(state)

    assert len(result["parent_sections"]) == 1


@pytest.mark.asyncio
async def test_expand_node_caps_distinct_parents(monkeypatch):
    monkeypatch.setenv("GENERATION_MAX_PARENTS", "3")
    monkeypatch.setenv("SQLITE_DB_PATH", "test.db")

    # 5 children with 5 different parents
    chunks = [make_chunk(chunk_id=f"c{i}", parent_chunk_id=f"p{i}", paper_id="arxiv1", score=0.9 - i*0.05) for i in range(5)]

    async def fake_fetchone():
        return {"id": "px", "parent_chunk_id": None, "paper_id": "arxiv1", "paper_title": "A Paper", "content_type": "text", "media_url": None, "section_name": "S", "text": "T"}

    call_count = 0

    async def fake_execute(sql, params=None):
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value={"id": params[0] if params else "px", "parent_chunk_id": None, "paper_id": "arxiv1", "paper_title": "A Paper", "published_date": "2023-01-01T00:00:00Z", "abstract": "An abstract.", "content_type": "text", "media_url": None, "section_name": "S", "text": f"text_{call_count}"})
        return mock_cursor

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(side_effect=fake_execute)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with patch("aiosqlite.connect", return_value=mock_conn):
        from backend.graph.nodes.expand import expand_node
        state = make_state(reranked_children=chunks)
        result = await expand_node(state)

    assert len(result["parent_sections"]) <= 3


# ---------------------------------------------------------------------------
# Grade node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grade_node_high_score_skips_llm_sufficient(monkeypatch):
    monkeypatch.setenv("GRADE_HIGH_THRESHOLD", "0.6")
    monkeypatch.setenv("GRADE_LOW_THRESHOLD", "0.35")
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")

    mock_groq = AsyncMock()

    with patch("groq.AsyncGroq", return_value=mock_groq):
        from backend.graph.nodes.grade import grade_node
        state = make_state(reranked_children=[make_chunk(score=0.8)])
        result = await grade_node(state)

    assert result["grade_result"]["sufficient"] is True
    assert not mock_groq.chat.completions.create.called


@pytest.mark.asyncio
async def test_grade_node_low_score_skips_llm_insufficient(monkeypatch):
    monkeypatch.setenv("GRADE_HIGH_THRESHOLD", "0.6")
    monkeypatch.setenv("GRADE_LOW_THRESHOLD", "0.35")
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")

    mock_groq = AsyncMock()

    with patch("groq.AsyncGroq", return_value=mock_groq):
        from backend.graph.nodes.grade import grade_node
        state = make_state(reranked_children=[make_chunk(score=0.2)])
        result = await grade_node(state)

    assert result["grade_result"]["sufficient"] is False
    assert not mock_groq.chat.completions.create.called


@pytest.mark.asyncio
async def test_grade_node_borderline_calls_grading_model(monkeypatch):
    monkeypatch.setenv("GRADE_HIGH_THRESHOLD", "0.6")
    monkeypatch.setenv("GRADE_LOW_THRESHOLD", "0.35")
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")

    mock_message = MagicMock()
    mock_message.content = "yes. The chunk directly answers the question."
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_completion)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    with patch("groq.AsyncGroq", return_value=mock_groq_instance):
        from backend.graph.nodes.grade import grade_node
        state = make_state(reranked_children=[make_chunk(score=0.5)])
        result = await grade_node(state)

    assert mock_completions.create.called


@pytest.mark.asyncio
async def test_grade_node_returns_sufficient_or_insufficient_with_reasoning(monkeypatch):
    monkeypatch.setenv("GRADE_HIGH_THRESHOLD", "0.6")
    monkeypatch.setenv("GRADE_LOW_THRESHOLD", "0.35")
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")

    mock_message = MagicMock()
    mock_message.content = "no. The chunk does not answer the question."
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_completion)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    with patch("groq.AsyncGroq", return_value=mock_groq_instance):
        from backend.graph.nodes.grade import grade_node
        state = make_state(reranked_children=[make_chunk(score=0.5)])
        result = await grade_node(state)

    gr = result["grade_result"]
    assert "sufficient" in gr
    assert "reasoning" in gr
    assert isinstance(gr["sufficient"], bool)
    assert isinstance(gr["reasoning"], str)


# ---------------------------------------------------------------------------
# Deep Dive branch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deep_dive_sufficient_routes_to_generate(monkeypatch):
    from backend.graph.edges import route_after_grade
    state = make_state(mode="deep_dive", grade_result={"sufficient": True, "reasoning": "high score"})
    result = route_after_grade(state)
    assert result == "generate_deep_dive"


@pytest.mark.asyncio
async def test_deep_dive_insufficient_does_not_call_ingest(monkeypatch):
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setenv("ASK_ARXIV_QUERY_COUNT", "3")

    mock_ingest = AsyncMock()

    mock_message = MagicMock()
    mock_message.content = "query1\nquery2\nquery3"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_completion)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    mock_http_response = MagicMock()
    mock_http_response.text = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Test Paper</title>
    <summary>An abstract about CRAG.</summary>
  </entry>
</feed>"""
    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(return_value=mock_http_response)
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    with patch("groq.AsyncGroq", return_value=mock_groq_instance), \
         patch("httpx.AsyncClient", return_value=mock_http_client), \
         patch("backend.graph.nodes.deep_dive_insufficient.hybrid_search", AsyncMock(return_value=[])), \
         patch("backend.graph.ingestion_graph.run_ingestion", mock_ingest):
        from backend.graph.nodes.deep_dive_insufficient import deep_dive_insufficient_node
        state = make_state(
            mode="deep_dive",
            user_message="Tell me about CRAG",
            retrieved_chunks=[make_chunk()],
            query_vector=[0.1, 0.2, 0.3],
        )
        await deep_dive_insufficient_node(state)

    assert not mock_ingest.called


@pytest.mark.asyncio
async def test_deep_dive_insufficient_generates_arxiv_queries_via_grading_model(monkeypatch):
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setenv("ASK_ARXIV_QUERY_COUNT", "3")

    model_used = []

    async def fake_create(**kwargs):
        model_used.append(kwargs.get("model"))
        mock_message = MagicMock()
        mock_message.content = "query1\nquery2\nquery3"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        return mock_completion

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(side_effect=fake_create)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    mock_http_response = MagicMock()
    mock_http_response.text = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Test Paper</title>
    <summary>An abstract about CRAG.</summary>
  </entry>
</feed>"""
    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(return_value=mock_http_response)
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    with patch("groq.AsyncGroq", return_value=mock_groq_instance), \
         patch("httpx.AsyncClient", return_value=mock_http_client), \
         patch("backend.graph.nodes.deep_dive_insufficient.hybrid_search", AsyncMock(return_value=[])):
        from backend.graph.nodes.deep_dive_insufficient import deep_dive_insufficient_node
        state = make_state(
            mode="deep_dive",
            user_message="Tell me about CRAG",
            retrieved_chunks=[make_chunk()],
            query_vector=[0.1, 0.2, 0.3],
        )
        await deep_dive_insufficient_node(state)

    assert any("llama-3.1-8b-instant" in (m or "") for m in model_used)


@pytest.mark.asyncio
async def test_deep_dive_insufficient_searches_arxiv_and_emits_citation_events(monkeypatch):
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")

    mock_message = MagicMock()
    mock_message.content = "corrective retrieval augmented generation"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_completion)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    mock_http_response = MagicMock()
    mock_http_response.text = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Test Paper Title</title>
    <summary>Abstract snippet here.</summary>
  </entry>
</feed>"""
    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(return_value=mock_http_response)
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    emitted: list[dict] = []

    with patch("groq.AsyncGroq", return_value=mock_groq_instance), \
         patch("httpx.AsyncClient", return_value=mock_http_client), \
         patch("backend.graph.nodes.deep_dive_insufficient.hybrid_search", AsyncMock(return_value=[])), \
         patch("backend.graph.nodes.deep_dive_insufficient.emit", emitted.append):
        from backend.graph.nodes.deep_dive_insufficient import deep_dive_insufficient_node
        state = make_state(
            mode="deep_dive", retrieved_chunks=[make_chunk()], query_vector=[0.1, 0.2, 0.3]
        )
        await deep_dive_insufficient_node(state)

    citation_events = [e for e in emitted if e.get("type") == "citation"]
    assert len(citation_events) >= 1
    assert "arxiv_id" in citation_events[0]
    assert "title" in citation_events[0]
    assert "abstract_snippet" in citation_events[0]


@pytest.mark.asyncio
async def test_deep_dive_insufficient_response_includes_not_enough_info_message(monkeypatch):
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")

    mock_message = MagicMock()
    mock_message.content = "some query"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_completion)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    mock_http_response = MagicMock()
    mock_http_response.text = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""
    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(return_value=mock_http_response)
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    emitted: list[dict] = []

    with patch("groq.AsyncGroq", return_value=mock_groq_instance), \
         patch("httpx.AsyncClient", return_value=mock_http_client), \
         patch("backend.graph.nodes.deep_dive_insufficient.hybrid_search", AsyncMock(return_value=[])), \
         patch("backend.graph.nodes.deep_dive_insufficient.emit", emitted.append):
        from backend.graph.nodes.deep_dive_insufficient import deep_dive_insufficient_node
        state = make_state(
            mode="deep_dive", retrieved_chunks=[make_chunk()], query_vector=[0.1, 0.2, 0.3]
        )
        await deep_dive_insufficient_node(state)

    token_events = [e for e in emitted if e.get("type") == "token"]
    all_tokens = "".join(e.get("content", "") for e in token_events)
    assert (
        "not enough" in all_tokens.lower()
        or "insufficient" in all_tokens.lower()
        or "don't have enough" in all_tokens.lower()
        or "do not have enough" in all_tokens.lower()
    )


@pytest.mark.asyncio
async def test_deep_dive_insufficient_does_not_write_sqlite(monkeypatch):
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")

    mock_message = MagicMock()
    mock_message.content = "query"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_completion)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    mock_http_response = MagicMock()
    mock_http_response.text = "<feed xmlns='http://www.w3.org/2005/Atom'></feed>"
    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(return_value=mock_http_response)
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    mock_aiosqlite_connect = AsyncMock()

    with patch("groq.AsyncGroq", return_value=mock_groq_instance), \
         patch("httpx.AsyncClient", return_value=mock_http_client), \
         patch("backend.graph.nodes.deep_dive_insufficient.hybrid_search", AsyncMock(return_value=[])), \
         patch("aiosqlite.connect", mock_aiosqlite_connect):
        from backend.graph.nodes.deep_dive_insufficient import deep_dive_insufficient_node
        state = make_state(
            mode="deep_dive", retrieved_chunks=[make_chunk()], query_vector=[0.1, 0.2, 0.3]
        )
        await deep_dive_insufficient_node(state)

    assert not mock_aiosqlite_connect.called


# ---------------------------------------------------------------------------
# Ask branch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_sufficient_routes_to_generate_with_citations():
    from backend.graph.edges import route_after_grade
    state = make_state(mode="ask", grade_result={"sufficient": True, "reasoning": "high score"})
    result = route_after_grade(state)
    assert result == "generate_ask"


@pytest.mark.asyncio
async def test_ask_insufficient_streams_searching_status_before_corrective_step(monkeypatch):
    monkeypatch.setenv("ASK_ARXIV_QUERY_COUNT", "3")
    monkeypatch.setenv("ASK_INGEST_CANDIDATES", "3")
    monkeypatch.setenv("CRAG_MAX_RETRIES", "2")
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")

    mock_message = MagicMock()
    mock_message.content = "q1\nq2\nq3"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_completion)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    mock_http_response = MagicMock()
    mock_http_response.text = "<feed xmlns='http://www.w3.org/2005/Atom'></feed>"
    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(return_value=mock_http_response)
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    mock_ingest = AsyncMock()

    emitted: list[dict] = []

    with patch("groq.AsyncGroq", return_value=mock_groq_instance), \
         patch("httpx.AsyncClient", return_value=mock_http_client), \
         patch("backend.graph.nodes.ask_corrective.run_ingestion", mock_ingest), \
         patch("backend.graph.nodes.ask_corrective.emit", emitted.append):
        from backend.graph.nodes.ask_corrective import ask_corrective_node
        state = make_state(mode="ask", retry_count=0)
        await ask_corrective_node(state)

    status_events = [
        e for e in emitted
        if e.get("type") == "status" and e.get("step") == "searching_arxiv"
    ]
    assert len(status_events) >= 1
    assert status_events[0]["content"] == "Looking up papers on arXiv…"


@pytest.mark.asyncio
async def test_ask_insufficient_generates_three_diverse_arxiv_queries(monkeypatch):
    monkeypatch.setenv("ASK_ARXIV_QUERY_COUNT", "3")
    monkeypatch.setenv("ASK_INGEST_CANDIDATES", "3")
    monkeypatch.setenv("CRAG_MAX_RETRIES", "2")
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")

    mock_message = MagicMock()
    mock_message.content = "query one\nquery two\nquery three"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_completion)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    mock_http_response = MagicMock()
    mock_http_response.text = "<feed xmlns='http://www.w3.org/2005/Atom'></feed>"
    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(return_value=mock_http_response)
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    mock_ingest = AsyncMock()

    with patch("groq.AsyncGroq", return_value=mock_groq_instance), \
         patch("httpx.AsyncClient", return_value=mock_http_client), \
         patch("backend.graph.nodes.ask_corrective.run_ingestion", mock_ingest):
        from backend.graph.nodes.ask_corrective import ask_corrective_node
        state = make_state(mode="ask", retry_count=0)
        result = await ask_corrective_node(state)

    assert len(result.get("arxiv_queries", [])) == 3


@pytest.mark.asyncio
async def test_ask_insufficient_searches_arxiv_concurrently_for_all_queries(monkeypatch):
    monkeypatch.setenv("ASK_ARXIV_QUERY_COUNT", "3")
    monkeypatch.setenv("ASK_INGEST_CANDIDATES", "3")
    monkeypatch.setenv("CRAG_MAX_RETRIES", "2")
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")

    mock_message = MagicMock()
    mock_message.content = "q1\nq2\nq3"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_completion)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    get_call_count = 0
    mock_http_response = MagicMock()
    mock_http_response.text = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.0000{i}v1</id>
    <title>Paper</title>
    <summary>Abstract.</summary>
  </entry>
</feed>"""
    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(return_value=mock_http_response)
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    mock_ingest = AsyncMock()

    with patch("groq.AsyncGroq", return_value=mock_groq_instance), \
         patch("httpx.AsyncClient", return_value=mock_http_client), \
         patch("backend.graph.nodes.ask_corrective.run_ingestion", mock_ingest):
        from backend.graph.nodes.ask_corrective import ask_corrective_node
        state = make_state(mode="ask", retry_count=0)
        await ask_corrective_node(state)

    # Should have called get 3 times (once per query)
    assert mock_http_client.get.call_count == 3


@pytest.mark.asyncio
async def test_ask_insufficient_calls_ingest_paper_for_relevant_candidates(monkeypatch):
    monkeypatch.setenv("ASK_ARXIV_QUERY_COUNT", "3")
    monkeypatch.setenv("ASK_INGEST_CANDIDATES", "2")
    monkeypatch.setenv("CRAG_MAX_RETRIES", "2")
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")

    call_count = [0]

    async def fake_create(**kwargs):
        call_count[0] += 1
        mock_message = MagicMock()
        if call_count[0] == 1:
            # First call: query generation
            mock_message.content = "q1\nq2\nq3"
        else:
            # Second call: relevance check — return relevant arxiv IDs
            mock_message.content = "2301.00001, 2301.00002"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        return mock_completion

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(side_effect=fake_create)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    mock_http_response = MagicMock()
    mock_http_response.text = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Paper A</title>
    <summary>Abstract A.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2301.00002v1</id>
    <title>Paper B</title>
    <summary>Abstract B.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2301.00003v1</id>
    <title>Paper C</title>
    <summary>Abstract C.</summary>
  </entry>
</feed>"""
    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(return_value=mock_http_response)
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    mock_ingest = AsyncMock()

    with patch("groq.AsyncGroq", return_value=mock_groq_instance), \
         patch("httpx.AsyncClient", return_value=mock_http_client), \
         patch("backend.graph.nodes.ask_corrective.run_ingestion", mock_ingest):
        from backend.graph.nodes.ask_corrective import ask_corrective_node
        state = make_state(mode="ask", retry_count=0)
        await ask_corrective_node(state)

    # GRADING_MODEL should have been called twice: once for query generation, once for relevance check
    assert mock_completions.create.call_count >= 2
    # Only relevant papers should be ingested (up to ASK_INGEST_CANDIDATES=2)
    assert mock_ingest.called
    assert mock_ingest.call_count <= 2


@pytest.mark.asyncio
async def test_ask_insufficient_reretrieves_after_ingestion(monkeypatch):
    """route_after_corrective should return 'retrieve' when retry_count < CRAG_MAX_RETRIES."""
    monkeypatch.setenv("CRAG_MAX_RETRIES", "2")
    from backend.graph.edges import route_after_corrective
    state = make_state(retry_count=1)  # < 2
    result = route_after_corrective(state)
    assert result == "retrieve"


@pytest.mark.asyncio
async def test_ask_corrective_loop_respects_crag_max_retries(monkeypatch):
    """route_after_corrective should return 'generate_ask' when retry_count >= CRAG_MAX_RETRIES."""
    monkeypatch.setenv("CRAG_MAX_RETRIES", "2")
    from backend.graph.edges import route_after_corrective
    state = make_state(retry_count=2)  # >= 2
    result = route_after_corrective(state)
    assert result == "generate_ask"


# ---------------------------------------------------------------------------
# Generation / streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_node_uses_generation_model_not_grading_model(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")

    models_used = []

    async def fake_stream(**kwargs):
        models_used.append(kwargs.get("model"))
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "Hello"

        class FakeStream:
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not hasattr(self, '_done'):
                    self._done = True
                    return chunk
                raise StopAsyncIteration
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        return FakeStream()

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(side_effect=fake_stream)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    with patch("groq.AsyncGroq", return_value=mock_groq_instance):
        from backend.graph.nodes.generate import generate_node
        state = make_state(parent_sections=[{"section_id": "p1", "text": "context", "paper_id": "arxiv1"}])
        await generate_node(state)

    assert "llama-3.3-70b-versatile" in models_used
    assert "llama-3.1-8b-instant" not in models_used


def _fake_groq_stream(content="Hello"):
    """Build a mock groq.AsyncGroq whose stream yields a single token."""
    async def fake_stream(**kwargs):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = content

        class FakeStream:
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not hasattr(self, "_done"):
                    self._done = True
                    return chunk
                raise StopAsyncIteration
        return FakeStream()

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(side_effect=fake_stream)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    instance = MagicMock()
    instance.chat = mock_chat
    return instance


@pytest.mark.asyncio
async def test_generate_node_emits_deduped_citations(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "m")
    import backend.graph.nodes.generate as gen

    # Citations now read the abstract carried on parent_sections (from expand),
    # so there is no DB lookup here at all.
    emitted: list[dict] = []
    state = make_state(
        intent="research",
        parent_sections=[
            {"paper_id": "2402.17764", "paper_title": "BitNet", "abstract": "BitNet abstract"},
            {"paper_id": "2402.17764", "paper_title": "BitNet", "abstract": "BitNet abstract"},
            {"paper_id": "2310.06825", "paper_title": "Mistral 7B", "abstract": "Mistral abstract"},
        ],
    )
    with patch("groq.AsyncGroq", return_value=_fake_groq_stream()), \
         patch("backend.graph.nodes.generate.emit", emitted.append):
        await gen.generate_node(state)

    cites = [e for e in emitted if e.get("type") == "citation"]
    assert [c["arxiv_id"] for c in cites] == ["2402.17764", "2310.06825"]
    assert cites[0]["title"] == "BitNet"
    assert cites[0]["abstract_snippet"] == "BitNet abstract"


@pytest.mark.asyncio
async def test_generate_node_chat_intent_emits_no_citations(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "m")
    import backend.graph.nodes.generate as gen

    emitted: list[dict] = []
    state = make_state(intent="chat", parent_sections=[])
    with patch("groq.AsyncGroq", return_value=_fake_groq_stream()), \
         patch("backend.graph.nodes.generate.emit", emitted.append):
        await gen.generate_node(state)

    assert not [e for e in emitted if e.get("type") == "citation"]


@pytest.mark.asyncio
async def test_deep_dive_insufficient_zero_chunks_emits_drift_message_no_arxiv(monkeypatch):
    # A Deep Dive that retrieved nothing means the paper isn't indexed (drift):
    # emit an honest message and do NOT run a generic arXiv search.
    async def _fail_groq(*a, **k):
        raise AssertionError("should not call the LLM on the drift path")

    def _fail_http(*a, **k):
        raise AssertionError("should not hit arXiv on the drift path")

    emitted: list[dict] = []
    with patch("backend.graph.nodes.deep_dive_insufficient.groq_complete", _fail_groq), \
         patch("httpx.AsyncClient", _fail_http), \
         patch("backend.graph.nodes.deep_dive_insufficient.emit", emitted.append):
        from backend.graph.nodes.deep_dive_insufficient import deep_dive_insufficient_node
        state = make_state(mode="deep_dive", retrieved_chunks=[])
        result = await deep_dive_insufficient_node(state)

    assert result == {"citations": []}
    assert not [e for e in emitted if e.get("type") == "citation"]
    tokens = "".join(e.get("content", "") for e in emitted if e.get("type") == "token")
    assert "out of sync" in tokens.lower()


# ---------------------------------------------------------------------------
# Deep Dive insufficient — corpus-first suggestions
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _fake_papers_conn(rows: list[dict]):
    """Minimal async aiosqlite stand-in for the papers title/abstract lookup."""
    class _Cursor:
        def __aiter__(self):
            async def gen():
                for r in rows:
                    yield r
            return gen()

    class _Conn:
        row_factory = None

        async def execute(self, *a, **k):
            return _Cursor()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    return MagicMock(return_value=_Conn())


@pytest.mark.asyncio
async def test_deep_dive_insufficient_suggests_from_corpus_first(monkeypatch):
    # Cross-corpus retrieval finds related ingested papers -> suggest those, no arXiv call.
    def _fail_http(*a, **k):
        raise AssertionError("arXiv must not be called when the corpus has suggestions")

    corpus_hits = [
        SimpleNamespace(paper_id="otherA"),
        SimpleNamespace(paper_id="otherB"),
    ]
    rows = [
        {"arxiv_id": "otherA", "title": "Paper A", "abstract": "Abstract A"},
        {"arxiv_id": "otherB", "title": "Paper B", "abstract": "Abstract B"},
    ]

    emitted: list[dict] = []
    with patch("backend.graph.nodes.deep_dive_insufficient.hybrid_search",
               AsyncMock(return_value=corpus_hits)), \
         patch("aiosqlite.connect", _fake_papers_conn(rows)), \
         patch("httpx.AsyncClient", _fail_http), \
         patch("backend.graph.nodes.deep_dive_insufficient.emit", emitted.append):
        from backend.graph.nodes.deep_dive_insufficient import deep_dive_insufficient_node
        state = make_state(
            mode="deep_dive",
            paper_id="current",
            retrieved_chunks=[make_chunk()],
            query_vector=[0.1, 0.2, 0.3],
        )
        result = await deep_dive_insufficient_node(state)

    cited_ids = [e["arxiv_id"] for e in emitted if e.get("type") == "citation"]
    assert cited_ids == ["otherA", "otherB"]
    assert result["citations"] == [
        {"arxiv_id": "otherA", "title": "Paper A"},
        {"arxiv_id": "otherB", "title": "Paper B"},
    ]


@pytest.mark.asyncio
async def test_deep_dive_insufficient_excludes_current_paper_from_corpus(monkeypatch):
    corpus_hits = [
        SimpleNamespace(paper_id="current"),
        SimpleNamespace(paper_id="current"),
        SimpleNamespace(paper_id="otherA"),
    ]
    rows = [{"arxiv_id": "otherA", "title": "Paper A", "abstract": "Abstract A"}]

    emitted: list[dict] = []
    with patch("backend.graph.nodes.deep_dive_insufficient.hybrid_search",
               AsyncMock(return_value=corpus_hits)), \
         patch("aiosqlite.connect", _fake_papers_conn(rows)), \
         patch("httpx.AsyncClient", lambda *a, **k: (_ for _ in ()).throw(
             AssertionError("arXiv must not be called"))), \
         patch("backend.graph.nodes.deep_dive_insufficient.emit", emitted.append):
        from backend.graph.nodes.deep_dive_insufficient import deep_dive_insufficient_node
        state = make_state(
            mode="deep_dive",
            paper_id="current",
            retrieved_chunks=[make_chunk()],
            query_vector=[0.1, 0.2, 0.3],
        )
        await deep_dive_insufficient_node(state)

    cited_ids = [e["arxiv_id"] for e in emitted if e.get("type") == "citation"]
    assert cited_ids == ["otherA"]
    assert "current" not in cited_ids


@pytest.mark.asyncio
async def test_deep_dive_insufficient_falls_back_to_arxiv_when_corpus_empty(monkeypatch):
    # Corpus has nothing else (only the deep-dived paper) -> fall back to arXiv search.
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")

    mock_message = MagicMock()
    mock_message.content = "crag query"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(return_value=mock_completion)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    mock_http_response = MagicMock()
    mock_http_response.text = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>ArXiv Fallback Paper</title>
    <summary>Fallback abstract.</summary>
  </entry>
</feed>"""
    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(return_value=mock_http_response)
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    emitted: list[dict] = []
    with patch("groq.AsyncGroq", return_value=mock_groq_instance), \
         patch("httpx.AsyncClient", return_value=mock_http_client), \
         patch("backend.graph.nodes.deep_dive_insufficient.hybrid_search", AsyncMock(return_value=[])), \
         patch("backend.graph.nodes.deep_dive_insufficient.emit", emitted.append):
        from backend.graph.nodes.deep_dive_insufficient import deep_dive_insufficient_node
        state = make_state(
            mode="deep_dive",
            paper_id="current",
            retrieved_chunks=[make_chunk()],
            query_vector=[0.1, 0.2, 0.3],
        )
        await deep_dive_insufficient_node(state)

    cited_ids = [e["arxiv_id"] for e in emitted if e.get("type") == "citation"]
    assert cited_ids == ["2301.00001"]


@pytest.mark.asyncio
async def test_classify_node_research_question_uses_llm(monkeypatch):
    import backend.graph.nodes.classify as classify_mod

    async def _research(*a, **k):
        return "RESEARCH"

    monkeypatch.setattr(classify_mod, "groq_complete", _research)
    result = await classify_mod.classify_node(
        {"user_message": "What is sliding window attention in transformers?"}
    )
    assert result == {"intent": "research"}


@pytest.mark.asyncio
async def test_classify_node_ignores_chat_substring_in_wordy_answer(monkeypatch):
    # A wordy classifier reply that merely mentions "chat" must not be misrouted to
    # the chat intent: only an exact one-word label counts.
    import backend.graph.nodes.classify as classify_mod

    async def _wordy(*a, **k):
        return "This is clearly a research question, not chat."

    monkeypatch.setattr(classify_mod, "groq_complete", _wordy)
    result = await classify_mod.classify_node(
        {"user_message": "Compare the two retrieval papers"}
    )
    assert result == {"intent": "research"}


@pytest.mark.asyncio
async def test_classify_node_exact_label_with_trailing_punctuation(monkeypatch):
    import backend.graph.nodes.classify as classify_mod

    async def _chat(*a, **k):
        return "CHAT."

    monkeypatch.setattr(classify_mod, "groq_complete", _chat)
    result = await classify_mod.classify_node({"user_message": "thanks so much"})
    assert result == {"intent": "chat"}


def _one_chunk_stream():
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = "ok"

    class FakeStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if not hasattr(self, "_done"):
                self._done = True
                return chunk
            raise StopAsyncIteration

    return FakeStream()


@pytest.mark.asyncio
async def test_generate_node_trims_context_to_token_budget(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GENERATION_MAX_CONTEXT_TOKENS", "50")

    sent = {}

    async def fake_create(**kwargs):
        sent["messages"] = kwargs["messages"]
        return _one_chunk_stream()

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(side_effect=fake_create)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    with patch("groq.AsyncGroq", return_value=mock_groq_instance):
        from backend.graph.nodes.generate import generate_node

        big = "lorem ipsum dolor sit amet " * 1000
        state = make_state(
            parent_sections=[
                {"section_id": "p1", "text": big, "paper_id": "a1", "section_title": "S"}
            ]
        )
        await generate_node(state)

    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    system_content = next(
        m["content"] for m in sent["messages"] if m["role"] == "system"
    )
    # Boilerplate prompt (incl. the injection guardrail) + a context capped near
    # the 50-token budget — nowhere near the thousands of tokens the raw section
    # would have contributed.
    assert len(enc.encode(system_content)) < 450


@pytest.mark.asyncio
async def test_generate_node_retries_with_smaller_context_on_groq_413(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "llama-3.3-70b-versatile")

    import httpx
    import groq

    calls: list[dict] = []

    def _make_413() -> groq.APIStatusError:
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(413, request=request)
        return groq.APIStatusError("Request too large", response=response, body=None)

    async def fake_create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise _make_413()
        return _one_chunk_stream()

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(side_effect=fake_create)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    with patch("groq.AsyncGroq", return_value=mock_groq_instance):
        from backend.graph.nodes.generate import generate_node

        big = "word " * 5000
        state = make_state(
            parent_sections=[{"section_id": "p1", "text": big, "paper_id": "a1"}]
        )
        await generate_node(state)

    assert len(calls) == 2  # 413 on the first attempt triggers one retry

    def _system_len(kwargs: dict) -> int:
        return len(next(m["content"] for m in kwargs["messages"] if m["role"] == "system"))

    # The retry rebuilds the prompt with a halved context budget, so it is smaller.
    assert _system_len(calls[1]) < _system_len(calls[0])


@pytest.mark.asyncio
async def test_generate_node_includes_passed_in_chat_history_in_prompt(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "llama-3.3-70b-versatile")

    messages_sent = []

    async def fake_stream(**kwargs):
        messages_sent.extend(kwargs.get("messages", []))
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "response"

        class FakeStream:
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not hasattr(self, '_done'):
                    self._done = True
                    return chunk
                raise StopAsyncIteration
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        return FakeStream()

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(side_effect=fake_stream)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    with patch("groq.AsyncGroq", return_value=mock_groq_instance):
        from backend.graph.nodes.generate import generate_node
        history = [{"role": "user", "content": "previous question"}, {"role": "assistant", "content": "previous answer"}]
        state = make_state(chat_history=history, parent_sections=[])
        await generate_node(state)

    all_content = " ".join(str(m) for m in messages_sent)
    assert "previous question" in all_content or any(
        m.get("content") == "previous question" for m in messages_sent if isinstance(m, dict)
    )


@pytest.mark.asyncio
async def test_generate_node_includes_parent_sections_with_section_and_source_labels(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "llama-3.3-70b-versatile")

    messages_sent = []

    async def fake_stream(**kwargs):
        messages_sent.extend(kwargs.get("messages", []))
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "response"

        class FakeStream:
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not hasattr(self, '_done'):
                    self._done = True
                    return chunk
                raise StopAsyncIteration
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        return FakeStream()

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(side_effect=fake_stream)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    with patch("groq.AsyncGroq", return_value=mock_groq_instance):
        from backend.graph.nodes.generate import generate_node
        parent_sections = [
            {"section_id": "s1", "text": "Section content here", "paper_id": "arxiv999", "section_title": "Introduction"},
        ]
        state = make_state(parent_sections=parent_sections)
        await generate_node(state)

    all_content = " ".join(
        m.get("content", "") for m in messages_sent if isinstance(m, dict)
    )
    assert "arxiv999" in all_content
    assert "Section content here" in all_content


@pytest.mark.asyncio
async def test_astream_chat_yields_token_events_then_done_event(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setenv("GRADE_HIGH_THRESHOLD", "0.6")
    monkeypatch.setenv("GRADE_LOW_THRESHOLD", "0.35")
    monkeypatch.setenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("JINA_MAX_CONCURRENCY", "4")
    monkeypatch.setenv("RERANK_CANDIDATE_COUNT", "30")
    monkeypatch.setenv("RERANK_TOP_K", "5")
    monkeypatch.setenv("GENERATION_MAX_PARENTS", "5")
    monkeypatch.setenv("SQLITE_DB_PATH", "test.db")

    # Jina rerank mock (query embedding is mocked via embed_query in the patch below)
    mock_rerank = AsyncMock(return_value={"results": [{"index": 0, "relevance_score": 0.9}]})

    # Setup groq mock for generate
    tokens = ["Hello", " world"]
    token_iter = iter(tokens)

    async def fake_stream(**kwargs):
        class FakeStream:
            def __init__(self):
                self._tokens = list(tokens)
                self._idx = 0
            def __aiter__(self):
                return self
            async def __anext__(self):
                if self._idx < len(self._tokens):
                    c = MagicMock()
                    c.choices = [MagicMock()]
                    c.choices[0].delta = MagicMock()
                    c.choices[0].delta.content = self._tokens[self._idx]
                    self._idx += 1
                    return c
                raise StopAsyncIteration
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
        return FakeStream()

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(side_effect=fake_stream)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    # High score chunk -> sufficient -> generate
    high_score_chunk = make_chunk(score=0.9)

    mock_hybrid = AsyncMock(return_value=[high_score_chunk])

    mock_cursor = AsyncMock()
    mock_cursor.fetchone = AsyncMock(return_value={"id": "p1", "parent_chunk_id": None, "paper_id": "arxiv1", "paper_title": "A Paper", "published_date": "2023-01-01T00:00:00Z", "abstract": "An abstract.", "content_type": "text", "media_url": None, "section_name": "S", "text": "context"})
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with patch("backend.graph.nodes.rerank._rerank_with_retry", mock_rerank), \
         patch("backend.graph.nodes.retrieve.embed_query", AsyncMock(return_value=[0.1, 0.2])), \
         patch("groq.AsyncGroq", return_value=mock_groq_instance), \
         patch("backend.graph.nodes.retrieve.hybrid_search", mock_hybrid), \
         patch("aiosqlite.connect", return_value=mock_conn):
        from backend.graph.entrypoint import astream_chat
        events = []
        async for event in astream_chat(mode="ask", paper_id=None, user_message="test", chat_history=[]):
            events.append(event)

    types = [e.get("type") for e in events]
    assert "token" in types
    assert types[-1] == "done"


@pytest.mark.asyncio
async def test_astream_chat_ask_demonstrative_clarifies_without_retrieval(monkeypatch):
    """An Ask-mode 'these papers' probe must clarify, not retrieve.

    The deterministic guard in classify_node routes straight to the clarify reply, so
    the stream should carry a 'clarifying' status step and emit no citations (no
    retrieval happened). hybrid_search is wired to explode to prove it's never called.
    """
    monkeypatch.setenv("GENERATION_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")

    tokens = ["Which", " topic", "?"]

    async def fake_stream(**kwargs):
        class FakeStream:
            def __init__(self):
                self._tokens = list(tokens)
                self._idx = 0
            def __aiter__(self):
                return self
            async def __anext__(self):
                if self._idx < len(self._tokens):
                    c = MagicMock()
                    c.choices = [MagicMock()]
                    c.choices[0].delta = MagicMock()
                    c.choices[0].delta.content = self._tokens[self._idx]
                    self._idx += 1
                    return c
                raise StopAsyncIteration
        return FakeStream()

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(side_effect=fake_stream)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    def _boom(*a, **k):
        raise AssertionError("clarify path must not retrieve")

    with patch("groq.AsyncGroq", return_value=mock_groq_instance), \
         patch("backend.graph.nodes.retrieve.hybrid_search", AsyncMock(side_effect=_boom)):
        from backend.graph.entrypoint import astream_chat
        events = []
        async for event in astream_chat(
            mode="ask",
            paper_id=None,
            user_message="what problem do these papers address and what methods do they use?",
            chat_history=[],
        ):
            events.append(event)

    steps = [e.get("step") for e in events if e.get("type") == "status"]
    assert "clarifying" in steps
    assert "retrieving" not in steps
    assert not any(e.get("type") == "citation" for e in events)
    assert "token" in [e.get("type") for e in events]


@pytest.mark.asyncio
async def test_astream_chat_does_not_write_to_sqlite(monkeypatch):
    """astream_chat should be read-only; only expand_node reads from SQLite, never writes."""
    monkeypatch.setenv("GENERATION_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setenv("GRADE_HIGH_THRESHOLD", "0.6")
    monkeypatch.setenv("GRADE_LOW_THRESHOLD", "0.35")
    monkeypatch.setenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("JINA_MAX_CONCURRENCY", "4")
    monkeypatch.setenv("RERANK_CANDIDATE_COUNT", "30")
    monkeypatch.setenv("RERANK_TOP_K", "5")
    monkeypatch.setenv("GENERATION_MAX_PARENTS", "5")
    monkeypatch.setenv("SQLITE_DB_PATH", "test.db")

    mock_rerank = AsyncMock(return_value={"results": [{"index": 0, "relevance_score": 0.9}]})

    tokens = ["response"]

    async def fake_stream(**kwargs):
        class FakeStream:
            def __init__(self):
                self._tokens = list(tokens)
                self._idx = 0
            def __aiter__(self):
                return self
            async def __anext__(self):
                if self._idx < len(self._tokens):
                    c = MagicMock()
                    c.choices = [MagicMock()]
                    c.choices[0].delta = MagicMock()
                    c.choices[0].delta.content = self._tokens[self._idx]
                    self._idx += 1
                    return c
                raise StopAsyncIteration
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
        return FakeStream()

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(side_effect=fake_stream)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    mock_hybrid = AsyncMock(return_value=[make_chunk(score=0.9)])

    execute_calls = []

    async def fake_execute(sql, params=None):
        execute_calls.append(sql.strip().upper())
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value={"id": "p1", "parent_chunk_id": None, "paper_id": "arxiv1", "paper_title": "A Paper", "published_date": "2023-01-01T00:00:00Z", "abstract": "An abstract.", "content_type": "text", "media_url": None, "section_name": "S", "text": "context"})
        return mock_cursor

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(side_effect=fake_execute)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with patch("backend.graph.nodes.rerank._rerank_with_retry", mock_rerank), \
         patch("backend.graph.nodes.retrieve.embed_query", AsyncMock(return_value=[0.1, 0.2])), \
         patch("groq.AsyncGroq", return_value=mock_groq_instance), \
         patch("backend.graph.nodes.retrieve.hybrid_search", mock_hybrid), \
         patch("aiosqlite.connect", return_value=mock_conn):
        from backend.graph.entrypoint import astream_chat
        async for _ in astream_chat(mode="ask", paper_id=None, user_message="test", chat_history=[]):
            pass

    # No INSERT/UPDATE/DELETE/CREATE should be in executed SQL
    for sql in execute_calls:
        assert not any(sql.startswith(w) for w in ["INSERT", "UPDATE", "DELETE", "CREATE", "DROP"]), \
            f"Unexpected write SQL: {sql}"


# ---------------------------------------------------------------------------
# Generate node — mode-specific system prompts (Fix 3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_node_deep_dive_uses_deep_research_prompt(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "llama-3.3-70b-versatile")

    messages_sent = []

    async def fake_stream(**kwargs):
        messages_sent.extend(kwargs.get("messages", []))
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "deep response"

        class FakeStream:
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not hasattr(self, '_done'):
                    self._done = True
                    return chunk
                raise StopAsyncIteration
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        return FakeStream()

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(side_effect=fake_stream)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    with patch("groq.AsyncGroq", return_value=mock_groq_instance):
        from backend.graph.nodes.generate import generate_node
        state = make_state(
            mode="deep_dive",
            parent_sections=[{"section_id": "s1", "text": "paper text", "paper_id": "arxiv42", "section_title": "Methods"}],
        )
        await generate_node(state)

    system_content = next(
        (m.get("content", "") for m in messages_sent if isinstance(m, dict) and m.get("role") == "system"),
        "",
    )
    assert "deep research assistant" in system_content.lower()
    assert "section title" in system_content.lower() or "section titles" in system_content.lower()


@pytest.mark.asyncio
async def test_generate_node_ask_uses_concise_prompt(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "llama-3.3-70b-versatile")

    messages_sent = []

    async def fake_stream(**kwargs):
        messages_sent.extend(kwargs.get("messages", []))
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "ask response"

        class FakeStream:
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not hasattr(self, '_done'):
                    self._done = True
                    return chunk
                raise StopAsyncIteration
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        return FakeStream()

    mock_completions = AsyncMock()
    mock_completions.create = AsyncMock(side_effect=fake_stream)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat = mock_chat

    with patch("groq.AsyncGroq", return_value=mock_groq_instance):
        from backend.graph.nodes.generate import generate_node
        state = make_state(mode="ask", parent_sections=[])
        await generate_node(state)

    system_content = next(
        (m.get("content", "") for m in messages_sent if isinstance(m, dict) and m.get("role") == "system"),
        "",
    )
    assert "concisely" in system_content.lower() or "multiple papers" in system_content.lower()


# ---------------------------------------------------------------------------
# Integration test skeletons (Fix 6)
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_full_deep_dive_turn_against_real_apis_sufficient_case():
    pytest.skip("Requires real Groq/Voyage/Qdrant/SQLite — run with -m integration")


@pytest.mark.integration
async def test_full_ask_turn_against_real_apis_insufficient_case():
    pytest.skip("Requires real Groq/Voyage/Qdrant/SQLite/arxiv — run with -m integration")
