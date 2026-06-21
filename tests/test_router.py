import pytest


@pytest.mark.asyncio
async def test_classify_clarify_ask_mode(monkeypatch):
    import backend.graph.nodes.classify as classify_mod

    async def _clarify(*a, **k):
        return "CLARIFY"

    monkeypatch.setattr(classify_mod, "groq_complete", _clarify)
    result = await classify_mod.classify_node(
        {"user_message": "tell me about the recent papers", "mode": "ask"}
    )
    assert result == {"intent": "clarify"}


@pytest.mark.asyncio
async def test_classify_general_ask_mode(monkeypatch):
    import backend.graph.nodes.classify as classify_mod

    async def _general(*a, **k):
        return "GENERAL"

    monkeypatch.setattr(classify_mod, "groq_complete", _general)
    result = await classify_mod.classify_node(
        {"user_message": "what is backpropagation?", "mode": "ask"}
    )
    assert result == {"intent": "general"}


@pytest.mark.asyncio
async def test_classify_research_ask_mode(monkeypatch):
    import backend.graph.nodes.classify as classify_mod

    async def _research(*a, **k):
        return "RESEARCH"

    monkeypatch.setattr(classify_mod, "groq_complete", _research)
    result = await classify_mod.classify_node(
        {"user_message": "compare the latest diffusion models", "mode": "ask"}
    )
    assert result == {"intent": "research"}


@pytest.mark.asyncio
async def test_classify_greeting_fast_path_no_llm_call(monkeypatch):
    import backend.graph.nodes.classify as classify_mod

    called = False

    async def _fail(*a, **k):
        nonlocal called
        called = True
        raise AssertionError("should not call the LLM for a greeting")

    monkeypatch.setattr(classify_mod, "groq_complete", _fail)
    result = await classify_mod.classify_node({"user_message": "hello there!", "mode": "ask"})
    assert result == {"intent": "chat"}
    assert called is False


@pytest.mark.asyncio
async def test_classify_deep_dive_coerces_clarify_to_research(monkeypatch):
    import backend.graph.nodes.classify as classify_mod

    async def _clarify(*a, **k):
        return "CLARIFY"

    monkeypatch.setattr(classify_mod, "groq_complete", _clarify)
    result = await classify_mod.classify_node(
        {"user_message": "help me", "mode": "deep_dive"}
    )
    assert result == {"intent": "research"}


@pytest.mark.asyncio
async def test_classify_deep_dive_coerces_general_to_research(monkeypatch):
    import backend.graph.nodes.classify as classify_mod

    async def _general(*a, **k):
        return "GENERAL"

    monkeypatch.setattr(classify_mod, "groq_complete", _general)
    result = await classify_mod.classify_node(
        {"user_message": "what is attention?", "mode": "deep_dive"}
    )
    assert result == {"intent": "research"}


def test_route_after_classify_maps_all_intents():
    from backend.graph.edges import route_after_classify

    assert route_after_classify({"intent": "clarify"}) == "clarify"
    assert route_after_classify({"intent": "general"}) == "general"
    assert route_after_classify({"intent": "chat"}) == "chat"
    assert route_after_classify({"intent": "research"}) == "retrieve"
    assert route_after_classify({}) == "retrieve"
