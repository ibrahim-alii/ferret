import logging
import os

from backend.graph.nodes._llm import groq_complete
from backend.graph.stream import emit

logger = logging.getLogger(__name__)


async def grade_node(state: dict) -> dict:
    emit({"type": "status", "step": "grading", "content": "Reading sections"})
    high_threshold = float(os.environ.get("GRADE_HIGH_THRESHOLD", "0.6"))
    low_threshold = float(os.environ.get("GRADE_LOW_THRESHOLD", "0.35"))
    grading_model = os.environ.get("GRADING_MODEL", "llama-3.1-8b-instant")

    mode = state.get("mode")
    children = state.get("reranked_children", [])
    top_score = children[0]["score"] if children else 0.0

    if top_score >= high_threshold:
        return {"grade_result": {"sufficient": True, "reasoning": "score above threshold"}}

    # In Deep Dive, broad/summary questions ("summarize this paper") inherently score
    # low against individual child chunks — no single chunk says "this paper is about X" —
    # so the score-only hard-fail wrongly refuses them. For deep_dive we instead fall
    # through to the LLM grader even in the low zone, letting it judge whether THIS
    # paper's chunk can address the question. This keeps the refusals intact: an
    # off-paper question (the retrieved chunk is about the wrong topic) or an off-topic
    # one ("tell a joke") still grades "no". Ask mode keeps the score-only hard-fail.
    if top_score <= low_threshold and mode != "deep_dive":
        return {"grade_result": {"sufficient": False, "reasoning": "score below threshold"}}

    top_chunk_text = children[0].get("text", "") if children else ""
    user_question = state.get("user_message", "")

    if mode == "deep_dive":
        system_prompt = (
            "You are a relevance grader for a single-paper Q&A session. The user is asking "
            "about one specific paper, and the chunk below is retrieved from that paper. "
            "Decide whether this paper can address the question. Answer 'yes' for questions "
            "about the paper itself — including broad ones like 'what is this paper about?' or "
            "'summarize this paper', for which any representative chunk of the paper suffices. "
            "Answer 'no' only when the question is about a different paper/topic that this "
            "chunk does not cover, or is unrelated to the paper (e.g. small talk or jokes). "
            "Reply with 'yes' or 'no' followed by a period and a brief reasoning."
        )
    else:
        system_prompt = (
            "You are a relevance grader. Given a user question and a retrieved chunk, "
            "determine if the chunk is relevant enough to answer the question. "
            "Reply with 'yes' or 'no' followed by a period and a brief reasoning."
        )

    content = await groq_complete(
        model=grading_model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Question: {user_question}\n\nChunk: {top_chunk_text}",
            },
        ],
        max_tokens=200,
    )

    response_text = content.strip().lower()
    sufficient = response_text.startswith("yes")
    reasoning = response_text

    return {"grade_result": {"sufficient": sufficient, "reasoning": reasoning}}
