import logging
import os

import groq
import tiktoken

from backend.graph.nodes._llm import get_groq_client
from backend.graph.stream import emit

logger = logging.getLogger(__name__)

# Token budgets keep the Groq request under the model's per-request / TPM limit.
# Free-tier llama-3.3-70b rejects oversized requests with HTTP 413 once the
# expanded small-to-big context (and a growing chat history) pile up. We bound
# both the context and the history, then retry with a smaller context budget if
# Groq still returns 413.
_DEFAULT_CONTEXT_TOKENS = 6000
_DEFAULT_HISTORY_TOKENS = 1500
_MIN_SECTION_TOKENS = 200  # don't bother including a section sliver smaller than this

_ENCODER: tiktoken.Encoding | None = None


def _encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def _count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    enc = _encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def _build_context(parent_sections: list[dict], budget_tokens: int) -> str:
    """Concatenate parent sections up to a token budget, truncating the last one."""
    parts: list[str] = []
    used = 0
    for section in parent_sections:
        section_title = section.get("section_title", "")
        paper_title = section.get("paper_title", "") or section.get("paper_id", "")
        text = section.get("text", "")
        # Label by human-readable paper title (not the arXiv ID) so the model cites
        # something the user actually recognizes.
        label = f"[Source: {paper_title}]" + (f" [{section_title}]" if section_title else "")
        block = f"{label}\n{text}"
        block_tokens = _count_tokens(block)
        if used + block_tokens <= budget_tokens:
            parts.append(block)
            used += block_tokens
            continue
        remaining = budget_tokens - used
        if remaining >= _MIN_SECTION_TOKENS:
            parts.append(_truncate_to_tokens(block, remaining))
        break
    return "\n\n".join(parts)


def _trim_history(chat_history: list[dict], budget_tokens: int) -> list[dict]:
    """Keep the most recent turns that fit within the history token budget."""
    trimmed: list[dict] = []
    used = 0
    for msg in reversed(chat_history):
        cost = _count_tokens(msg.get("content", ""))
        if used + cost > budget_tokens:
            break
        trimmed.append(msg)
        used += cost
    trimmed.reverse()
    return trimmed


def _build_messages(
    *,
    mode: str,
    intent: str | None,
    parent_sections: list[dict],
    chat_history: list[dict],
    user_message: str,
    context_budget: int,
    history_budget: int,
) -> list[dict]:
    if intent == "chat":
        # Conversational turn: answer directly, no retrieved context.
        system_prompt = (
            "You are Ferret, a friendly research-assistant chatbot for arXiv papers. "
            "Respond naturally and briefly to greetings and small talk. If the user asks "
            "what you can do, explain that you answer questions about research papers in "
            "Ask mode (across the corpus) or Deep Dive mode (a single paper)."
        )
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(_trim_history(chat_history, history_budget))
        messages.append({"role": "user", "content": user_message})
        return messages

    context = _build_context(parent_sections, context_budget)

    if mode == "deep_dive":
        system_prompt = (
            "You are a deep research assistant analyzing a specific paper. "
            "Answer the user's question in depth using the provided sections from the paper. "
            "Refer to papers by their title (never the arXiv ID) and cite section titles "
            "when referencing information."
        )
    else:
        system_prompt = (
            "You are a research assistant with access to multiple papers. "
            "Answer the user's question concisely using the provided context. "
            "Refer to papers by their title, not the arXiv ID, when referencing information."
        )
    if context:
        system_prompt += f"\n\nContext:\n{context}"

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(_trim_history(chat_history, history_budget))
    messages.append({"role": "user", "content": user_message})
    return messages


async def generate_node(state: dict) -> dict:
    generation_model = os.environ.get("GENERATION_MODEL", "llama-3.3-70b-versatile")
    context_budget = int(os.environ.get("GENERATION_MAX_CONTEXT_TOKENS", _DEFAULT_CONTEXT_TOKENS))
    history_budget = int(os.environ.get("GENERATION_MAX_HISTORY_TOKENS", _DEFAULT_HISTORY_TOKENS))

    parent_sections = state.get("parent_sections", [])
    chat_history = state.get("chat_history", [])
    user_message = state.get("user_message", "")
    mode = state.get("mode", "ask")
    intent = state.get("intent")

    emit({"type": "status", "step": "generating", "content": "Writing answer"})

    client = get_groq_client()

    # Up to 3 attempts: each Groq 413 ("request too large") halves the context
    # budget. We only retry while no tokens have been emitted, so the user never
    # sees a partial answer restart.
    for attempt in range(3):
        messages = _build_messages(
            mode=mode,
            intent=intent,
            parent_sections=parent_sections,
            chat_history=chat_history,
            user_message=user_message,
            context_budget=context_budget,
            history_budget=history_budget,
        )
        emitted = False
        try:
            stream = await client.chat.completions.create(
                model=generation_model,
                messages=messages,
                stream=True,
                max_tokens=1024,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    emitted = True
                    emit({"type": "token", "content": delta.content})
            break
        except groq.APIStatusError as exc:
            too_large = getattr(exc, "status_code", None) == 413
            if too_large and not emitted and attempt < 2:
                context_budget = max(context_budget // 2, _MIN_SECTION_TOKENS)
                logger.warning(
                    "Groq 413 (request too large); retrying with context budget %d",
                    context_budget,
                )
                continue
            raise

    emit({"type": "done"})

    return {}
