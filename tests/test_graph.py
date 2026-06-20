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
async def test_rerank_node_calls_voyage_rerank_with_top_n_candidates(monkeypatch):
    monkeypatch.setenv("VOYAGE_RERANK_MODEL", "rerank-2.5-lite")
    monkeypatch.setenv("RERANK_CANDIDATE_COUNT", "30")
    monkeypatch.setenv("RERANK_TOP_K", "5")

    mock_result = MagicMock()
    mock_result.results = [MagicMock(index=0, relevance_score=0.9)]
    mock_voyage = AsyncMock()
    mock_voyage.rerank = AsyncMock(return_value=mock_result)

    chunks = [make_chunk(chunk_id=f"c{i}") for i in range(30)]

    with patch("voyageai.AsyncClient", return_value=mock_voyage):
        from backend.graph.nodes.rerank import rerank_node
        state = make_state(retrieved_chunks=chunks)
        await rerank_node(state)

    assert mock_voyage.rerank.called
    call_kwargs = mock_voyage.rerank.call_args
    documents_passed = call_kwargs.kwargs.get("documents", None) or (call_kwargs.args[1] if len(call_kwargs.args) > 1 else None)
    assert documents_passed is not None
    assert len(documents_passed) == 30  # RERANK_CANDIDATE_COUNT


@pytest.mark.asyncio
async def test_rerank_node_returns_top_k(monkeypatch):
    monkeypatch.setenv("VOYAGE_RERANK_MODEL", "rerank-2.5-lite")
    monkeypatch.setenv("RERANK_CANDIDATE_COUNT", "30")
    monkeypatch.setenv("RERANK_TOP_K", "5")

    # Build 10 mock results sorted by score descending
    mock_results = [MagicMock(index=i, relevance_score=1.0 - i * 0.05) for i in range(10)]
    mock_result = MagicMock()
    mock_result.results = mock_results
    mock_voyage = AsyncMock()
    mock_voyage.rerank = AsyncMock(return_value=mock_result)

    chunks = [make_chunk(chunk_id=f"c{i}", text=f"text {i}") for i in range(10)]

    with patch("voyageai.AsyncClient", return_value=mock_voyage):
        from backend.graph.nodes.rerank import rerank_node
        state = make_state(retrieved_chunks=chunks)
        result = await rerank_node(state)

    assert len(result["reranked_children"]) == 5  # RERANK_TOP_K


# ---------------------------------------------------------------------------
# Expand node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_expand_node_fetches_parent_sections_for_reranked_children(monkeypatch):
    monkeypatch.setenv("GENERATION_MAX_PARENTS", "5")
    monkeypatch.setenv("SQLITE_DB_PATH", "test.db")

    chunks = [make_chunk(chunk_id="c1", parent_chunk_id="p1", paper_id="arxiv1", score=0.9)]

    mock_row = {"id": "p1", "parent_chunk_id": None, "paper_id": "arxiv1", "section_name": "Intro", "text": "Parent text"}

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


@pytest.mark.asyncio
async def test_expand_node_dedupes_shared_parents(monkeypatch):
    monkeypatch.setenv("GENERATION_MAX_PARENTS", "5")
    monkeypatch.setenv("SQLITE_DB_PATH", "test.db")

    # Two children sharing the same parent
    chunks = [
        make_chunk(chunk_id="c1", parent_chunk_id="p1", paper_id="arxiv1", score=0.9),
        make_chunk(chunk_id="c2", parent_chunk_id="p1", paper_id="arxiv1", score=0.8),
    ]

    mock_row = {"id": "p1", "parent_chunk_id": None, "paper_id": "arxiv1", "section_name": "Intro", "text": "Parent text"}

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
        return {"id": "px", "parent_chunk_id": None, "paper_id": "arxiv1", "section_name": "S", "text": "T"}

    call_count = 0

    async def fake_execute(sql, params=None):
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value={"id": params[0] if params else "px", "parent_chunk_id": None, "paper_id": "arxiv1", "section_name": "S", "text": f"text_{call_count}"})
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
         patch("backend.graph.ingestion_graph.run_ingestion", mock_ingest):
        from backend.graph.nodes.deep_dive_insufficient import deep_dive_insufficient_node
        state = make_state(mode="deep_dive", user_message="Tell me about CRAG")
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
         patch("httpx.AsyncClient", return_value=mock_http_client):
        from backend.graph.nodes.deep_dive_insufficient import deep_dive_insufficient_node
        state = make_state(mode="deep_dive", user_message="Tell me about CRAG")
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
         patch("backend.graph.nodes.deep_dive_insufficient.emit", emitted.append):
        from backend.graph.nodes.deep_dive_insufficient import deep_dive_insufficient_node
        state = make_state(mode="deep_dive")
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
         patch("backend.graph.nodes.deep_dive_insufficient.emit", emitted.append):
        from backend.graph.nodes.deep_dive_insufficient import deep_dive_insufficient_node
        state = make_state(mode="deep_dive")
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
         patch("aiosqlite.connect", mock_aiosqlite_connect):
        from backend.graph.nodes.deep_dive_insufficient import deep_dive_insufficient_node
        state = make_state(mode="deep_dive")
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
async def test_ask_insufficient_streams_interim_message_before_corrective_step(monkeypatch):
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

    interim_events = [e for e in emitted if e.get("type") == "interim_message"]
    assert len(interim_events) >= 1


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
    monkeypatch.setenv("VOYAGE_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("RERANK_CANDIDATE_COUNT", "30")
    monkeypatch.setenv("RERANK_TOP_K", "5")
    monkeypatch.setenv("GENERATION_MAX_PARENTS", "5")
    monkeypatch.setenv("SQLITE_DB_PATH", "test.db")

    # Voyage rerank mock (query embedding is mocked via embed_query in the patch below)
    mock_voyage = AsyncMock()
    rerank_result = MagicMock()
    rerank_result.results = [MagicMock(index=0, relevance_score=0.9)]
    mock_voyage.rerank = AsyncMock(return_value=rerank_result)

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
    mock_cursor.fetchone = AsyncMock(return_value={"id": "p1", "parent_chunk_id": None, "paper_id": "arxiv1", "section_name": "S", "text": "context"})
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_cursor)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with patch("voyageai.AsyncClient", return_value=mock_voyage), \
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
async def test_astream_chat_does_not_write_to_sqlite(monkeypatch):
    """astream_chat should be read-only; only expand_node reads from SQLite, never writes."""
    monkeypatch.setenv("GENERATION_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GRADING_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setenv("GRADE_HIGH_THRESHOLD", "0.6")
    monkeypatch.setenv("GRADE_LOW_THRESHOLD", "0.35")
    monkeypatch.setenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("VOYAGE_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("RERANK_CANDIDATE_COUNT", "30")
    monkeypatch.setenv("RERANK_TOP_K", "5")
    monkeypatch.setenv("GENERATION_MAX_PARENTS", "5")
    monkeypatch.setenv("SQLITE_DB_PATH", "test.db")

    mock_voyage = AsyncMock()
    rerank_result = MagicMock()
    rerank_result.results = [MagicMock(index=0, relevance_score=0.9)]
    mock_voyage.rerank = AsyncMock(return_value=rerank_result)

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
        mock_cursor.fetchone = AsyncMock(return_value={"id": "p1", "parent_chunk_id": None, "paper_id": "arxiv1", "section_name": "S", "text": "context"})
        return mock_cursor

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(side_effect=fake_execute)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    with patch("voyageai.AsyncClient", return_value=mock_voyage), \
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
