"""
Tests for generate.py intent-specific prompts and citation gating.

The Groq client is mocked; emitted events are captured by patching
backend.graph.nodes.generate.emit. _build_messages is a plain function and is
unit-tested directly without mocking Groq.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def make_state(**overrides):
    base = {
        "mode": "ask",
        "paper_id": None,
        "user_message": "Tell me about transformers",
        "chat_history": [],
        "parent_sections": [],
        "intent": None,
    }
    base.update(overrides)
    return base


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


def _build(**overrides):
    from backend.graph.nodes.generate import _build_messages

    kwargs = {
        "mode": "ask",
        "intent": None,
        "parent_sections": [],
        "chat_history": [],
        "user_message": "Tell me about transformers",
        "context_budget": 6000,
        "history_budget": 1500,
    }
    kwargs.update(overrides)
    return _build_messages(**kwargs)


# ---------------------------------------------------------------------------
# Streaming: clarify / general emit no citations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_node_clarify_intent_emits_no_citations(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "m")
    import backend.graph.nodes.generate as gen

    emitted: list[dict] = []
    state = make_state(intent="clarify", parent_sections=[])
    with patch("groq.AsyncGroq", return_value=_fake_groq_stream()), \
         patch("backend.graph.nodes.generate.emit", emitted.append):
        await gen.generate_node(state)

    assert [e for e in emitted if e.get("type") == "token"]
    assert not [e for e in emitted if e.get("type") == "citation"]


@pytest.mark.asyncio
async def test_generate_node_general_intent_emits_no_citations(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "m")
    import backend.graph.nodes.generate as gen

    emitted: list[dict] = []
    state = make_state(intent="general", parent_sections=[])
    with patch("groq.AsyncGroq", return_value=_fake_groq_stream()), \
         patch("backend.graph.nodes.generate.emit", emitted.append):
        await gen.generate_node(state)

    assert [e for e in emitted if e.get("type") == "token"]
    assert not [e for e in emitted if e.get("type") == "citation"]


# ---------------------------------------------------------------------------
# _build_messages prompt content
# ---------------------------------------------------------------------------

def test_clarify_prompt_asks_clarifying_question():
    messages = _build(intent="clarify")
    system = messages[0]["content"].lower()
    assert "clarifying question" in system
    assert "do not" in system  # do NOT answer yet


def test_general_prompt_forbids_fabrication_and_notes_general_knowledge():
    messages = _build(intent="general")
    system = messages[0]["content"].lower()
    assert "never fabricate" in system
    assert "general background" in system or "general or conceptual" in system


def test_ask_research_prompt_permits_general_knowledge_supplementation():
    messages = _build(mode="ask", intent="research")
    system = messages[0]["content"].lower()
    assert "only the provided context" not in system
    assert "supplement with general knowledge" in system


def test_clarify_and_general_include_history_and_user_message():
    history = [{"role": "user", "content": "earlier turn"}]
    for intent in ("clarify", "general"):
        messages = _build(intent=intent, chat_history=history)
        assert messages[0]["role"] == "system"
        assert {"role": "user", "content": "earlier turn"} in messages
        assert messages[-1] == {"role": "user", "content": "Tell me about transformers"}
