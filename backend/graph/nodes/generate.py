import logging
import os

from backend.graph.nodes._llm import get_groq_client
from backend.graph.stream import emit

logger = logging.getLogger(__name__)


async def generate_node(state: dict) -> dict:
    generation_model = os.environ.get("GENERATION_MODEL", "llama-3.3-70b-versatile")

    parent_sections = state.get("parent_sections", [])
    chat_history = state.get("chat_history", [])
    user_message = state.get("user_message", "")

    # Build context from parent sections
    context_parts = []
    for section in parent_sections:
        title = section.get("section_title", "")
        paper_id = section.get("paper_id", "")
        text = section.get("text", "")
        label = f"[Source: {paper_id}]" + (f" [{title}]" if title else "")
        context_parts.append(f"{label}\n{text}")

    context = "\n\n".join(context_parts)

    mode = state.get("mode", "ask")
    if mode == "deep_dive":
        system_prompt = (
            "You are a deep research assistant analyzing a specific paper. "
            "Answer the user's question in depth using the provided sections from the paper. "
            "Cite the source paper ID and section titles when referencing information."
        )
    else:
        system_prompt = (
            "You are a research assistant with access to multiple papers. "
            "Answer the user's question concisely using the provided context. "
            "Cite source paper IDs when referencing information."
        )
    if context:
        system_prompt += f"\n\nContext:\n{context}"

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    emit({"type": "status", "step": "generating", "content": "Writing answer"})

    client = get_groq_client()
    stream = await client.chat.completions.create(
        model=generation_model,
        messages=messages,
        stream=True,
        max_tokens=1024,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            emit({"type": "token", "content": delta.content})

    emit({"type": "done"})

    return {}
