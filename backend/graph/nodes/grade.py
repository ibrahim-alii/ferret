import logging
import os

import groq

logger = logging.getLogger(__name__)


async def grade_node(state: dict) -> dict:
    high_threshold = float(os.environ.get("GRADE_HIGH_THRESHOLD", "0.6"))
    low_threshold = float(os.environ.get("GRADE_LOW_THRESHOLD", "0.35"))
    grading_model = os.environ.get("GRADING_MODEL", "llama-3.1-8b-instant")

    children = state.get("reranked_children", [])
    top_score = children[0]["score"] if children else 0.0

    if top_score >= high_threshold:
        return {"grade_result": {"sufficient": True, "reasoning": "score above threshold"}}

    if top_score <= low_threshold:
        return {"grade_result": {"sufficient": False, "reasoning": "score below threshold"}}

    # Borderline: call LLM
    top_chunk_text = children[0].get("text", "") if children else ""
    user_question = state.get("user_message", "")

    client = groq.AsyncGroq()
    completion = await client.chat.completions.create(
        model=grading_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a relevance grader. Given a user question and a retrieved chunk, "
                    "determine if the chunk is relevant enough to answer the question. "
                    "Reply with 'yes' or 'no' followed by a period and a brief reasoning."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {user_question}\n\nChunk: {top_chunk_text}",
            },
        ],
        max_tokens=200,
    )

    response_text = completion.choices[0].message.content.strip().lower()
    sufficient = response_text.startswith("yes")
    reasoning = response_text

    return {"grade_result": {"sufficient": sufficient, "reasoning": reasoning}}
