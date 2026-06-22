"""Maximal Marginal Relevance de-duplication for Ask mode.

Ask searches the whole corpus, so similar papers surface near-identical chunks.
After reranking, MMR trades a little relevance for diversity to drop those
duplicates before the context is built. Deep Dive (single paper) is left as-is.
"""
import logging
import math
import os

logger = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _mmr_select(chunks: list[dict], lambda_: float, top_k: int) -> list[dict]:
    """Greedy MMR: relevance = rerank score, similarity = cosine of dense vectors."""
    selected: list[dict] = []
    candidates = list(chunks)
    while candidates and len(selected) < top_k:
        best_idx = 0
        best_score = -math.inf
        for i, cand in enumerate(candidates):
            rel = cand.get("score", 0.0)
            if selected:
                max_sim = max(
                    _cosine(cand["dense_vector"], s["dense_vector"]) for s in selected
                )
            else:
                max_sim = 0.0
            mmr = lambda_ * rel - (1.0 - lambda_) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected.append(candidates.pop(best_idx))
    return selected


async def mmr_node(state: dict) -> dict:
    top_k = int(os.environ.get("RERANK_TOP_K", "5"))
    reranked = state.get("reranked_children", [])

    # Deep Dive is single-paper; MMR adds little, so just trim to the final top_k.
    if state.get("mode") != "ask":
        return {"reranked_children": reranked[:top_k]}

    query_vector = state.get("query_vector")
    # MMR needs every candidate's dense vector; if any is missing (e.g. an older
    # point upserted before vectors were returned), fall back to rerank order.
    if not query_vector or any(not c.get("dense_vector") for c in reranked):
        if reranked:
            logger.debug("MMR skipped: missing query or chunk vectors")
        return {"reranked_children": reranked[:top_k]}

    lambda_ = float(os.environ.get("MMR_LAMBDA", "0.7"))
    return {"reranked_children": _mmr_select(reranked, lambda_, top_k)}
