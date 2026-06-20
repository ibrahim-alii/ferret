import logging
import os
import re

from backend.graph.nodes._llm import groq_complete

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


async def classify_node(state: dict) -> dict:
    """Decide whether a turn is conversational ("chat") or a research question.

    Conversational turns skip the whole CRAG retrieval/rerank/corrective pipeline and
    are answered directly by the LLM, so "hi there!" doesn't trigger an arXiv search.
    """
    message = state.get("user_message", "").strip()

    if not message or _CHITCHAT_RE.match(message):
        return {"intent": "chat"}

    model = os.environ.get("GRADING_MODEL", "llama-3.1-8b-instant")
    try:
        answer = await groq_complete(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user's message as either CHAT or RESEARCH. "
                        "CHAT = greetings, small talk, thanks, or meta questions about the assistant. "
                        "RESEARCH = any question whose answer needs information from research papers. "
                        "Reply with exactly one word: CHAT or RESEARCH."
                    ),
                },
                {"role": "user", "content": message[:500]},
            ],
            max_tokens=3,
        )
    except Exception as exc:
        # On any classifier failure, default to the full research path (safer than
        # answering a real question with no grounding).
        logger.warning("intent classification failed (%s); defaulting to research", exc)
        return {"intent": "research"}

    intent = "chat" if "chat" in answer.strip().lower() else "research"
    return {"intent": intent}
