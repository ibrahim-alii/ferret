import logging
import os
import re

from backend.graph.nodes._llm import groq_complete
from backend.graph.stream import emit

logger = logging.getLogger(__name__)

# Obvious greetings / small talk that should never touch retrieval. Matched first so
# the common chitchat case is instant (no LLM call at all).
_CHITCHAT_RE = re.compile(
    r"^("
    r"(hi|hii+|hey+|hello|hiya|yo|sup|howdy|gm|gn)( there| everyone| all| folks| ferret)?"
    r"|good\s+(morning|afternoon|evening|night)"
    r"|thanks?|thank\s+you|thx|ty|ok(ay)?|cool|nice|great|awesome|lol|bye|goodbye"
    r"|how\s+are\s+you|who\s+are\s+you|what\s+can\s+you\s+do"
    r")[\s!.?,]*$",
    re.IGNORECASE,
)

# Demonstrative references to a specific-but-unnamed paper set. In Ask mode there is no
# fixed "these papers" in view (the corpus grows arbitrarily), so such a reference means
# the user is presuming a single/known-set context that only Deep Dive provides. Matched
# ask-mode-only and routed to clarify, which nudges the user toward Deep Dive.
_UNNAMED_PAPER_REF_RE = re.compile(
    r"\b(this|these|those|the\s+above|the\s+aforementioned)\s+papers?\b",
    re.IGNORECASE,
)


async def classify_node(state: dict) -> dict:
    """Route a turn to one of four intents: chat, clarify, general, or research.

    Conversational turns skip the whole CRAG retrieval/rerank/corrective pipeline and
    are answered directly by the LLM, so "hi there!" doesn't trigger an arXiv search.
    In deep_dive mode only "chat" and "research" are valid; clarify/general are
    ask-mode-only and are coerced to "research" under deep_dive.
    """
    message = state.get("user_message", "").strip()

    if not message or _CHITCHAT_RE.match(message):
        emit({"type": "status", "step": "chatting", "content": "Replying…"})
        return {"intent": "chat"}

    mode = state.get("mode", "ask")

    # Ask-mode-only guard: a demonstrative reference to an unnamed paper set has no
    # referent here (the corpus isn't a fixed set the user is looking at), so ask one
    # clarifying question instead of searching the whole corpus. Deterministic, so it
    # skips the LLM call entirely.
    if mode != "deep_dive" and _UNNAMED_PAPER_REF_RE.search(message):
        emit({"type": "status", "step": "clarifying", "content": "Clarifying…"})
        return {"intent": "clarify"}

    model = os.environ.get("GRADING_MODEL", "llama-3.1-8b-instant")
    if mode == "deep_dive":
        system_prompt = (
            "Classify the user's message as either CHAT or RESEARCH. "
            "CHAT = greetings, small talk, thanks, or meta questions about the assistant. "
            "RESEARCH = anything else, i.e. any question that benefits from looking at the paper. "
            "Reply with exactly one word: CHAT or RESEARCH."
        )
    else:
        system_prompt = (
            "Classify the user's message as exactly one of CHAT, CLARIFY, GENERAL, or RESEARCH. "
            "CHAT = greetings, small talk, thanks, or meta questions about the assistant. "
            "CLARIFY = the request is too vague, ambiguous, or underspecified to search or "
            "answer well (e.g. 'tell me about the recent papers' with no topic, 'help me', "
            "'what's interesting'), OR it refers to a specific but unnamed set of papers the "
            "user seems to be looking at ('these papers', 'this paper') without naming a "
            "topic or arXiv id. "
            "GENERAL = a general-knowledge or conceptual question not tied to retrieving specific "
            "papers (e.g. 'what is backpropagation?', 'explain attention'). "
            "RESEARCH = anything that benefits from searching arXiv for specific papers: content "
            "questions about recent or latest papers on a named topic, comparisons, or "
            "summaries of a named area. "
            "Reply with exactly one word: CHAT, CLARIFY, GENERAL, or RESEARCH."
        )
    try:
        answer = await groq_complete(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message[:500]},
            ],
            max_tokens=3,
        )
    except Exception as exc:
        # On any classifier failure, default to the full research path (safer than
        # answering a real question with no grounding).
        logger.warning("intent classification failed (%s); defaulting to research", exc)
        return {"intent": "research"}

    # The model is told to reply with exactly one label, but it sometimes wraps it
    # ("CHAT.") or pads it into a sentence. Match the bare label only — a substring
    # check misroutes wordy replies like "not chat" to the chat intent. Anything that
    # isn't an exact label falls through to the safe research default.
    lowered = answer.strip().lower().strip(".!?,'\" ")
    if lowered in ("chat", "clarify", "general"):
        intent = lowered
    else:
        intent = "research"

    # clarify/general are ask-mode-only; under deep_dive fall back to research.
    if mode == "deep_dive" and intent in ("clarify", "general"):
        intent = "research"

    if intent == "chat":
        emit({"type": "status", "step": "chatting", "content": "Replying…"})
    elif intent == "clarify":
        emit({"type": "status", "step": "clarifying", "content": "Clarifying…"})
    elif intent == "general":
        emit({"type": "status", "step": "answering", "content": "Answering…"})
    return {"intent": intent}
