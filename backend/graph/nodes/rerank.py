import asyncio
import logging
import os

import httpx

from backend.graph.stream import emit

logger = logging.getLogger(__name__)

_JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"

# Shared across requests so JINA_MAX_CONCURRENCY bounds total in-flight rerank calls.
_semaphore = asyncio.Semaphore(int(os.environ.get("JINA_MAX_CONCURRENCY", "4")))

_MAX_RETRIES = 3
# HTTP status codes worth retrying: rate limit + transient upstream errors.
_RETRY_STATUS = {429, 500, 502, 503, 504}


async def _rerank_with_retry(payload: dict, headers: dict) -> dict:
    """Call Jina's rerank endpoint with retry/backoff on transient errors.

    Mirrors the retry shape of _llm.groq_complete. Transient = network errors or a
    retryable HTTP status; everything else (e.g. 401/422) raises immediately.
    """
    base_wait = int(os.environ.get("JINA_RETRY_BASE_WAIT", "2"))
    timeout = float(os.environ.get("JINA_TIMEOUT", "30"))
    for attempt in range(_MAX_RETRIES):
        try:
            async with _semaphore:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(_JINA_RERANK_URL, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            retriable = isinstance(exc, httpx.TransportError) or (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response.status_code in _RETRY_STATUS
            )
            if not retriable or attempt == _MAX_RETRIES - 1:
                raise
            wait = base_wait * (attempt + 1)
            logger.warning("Jina rerank failed (%s); retrying in %ds", exc, wait)
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


async def rerank_node(state: dict) -> dict:
    emit({"type": "status", "step": "reranking", "content": "Ranking results"})
    rerank_model = os.environ.get("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")
    top_k = int(os.environ.get("RERANK_TOP_K", "5"))

    # Jina rejects empty strings and empty document lists. Drop chunks with no text
    # (e.g. a Qdrant hit whose chunk id has no matching SQLite row) — they carry no
    # signal to rerank on and would otherwise abort the whole request.
    chunks = [c for c in state["retrieved_chunks"] if c.get("text", "").strip()]
    if not chunks:
        if state["retrieved_chunks"]:
            # Hits came back from Qdrant but none had SQLite text — a sign the two
            # stores are out of sync. Surface it so it's diagnosable from logs.
            logger.warning(
                "All %d retrieved chunks have empty text; check SQLite/Qdrant sync",
                len(state["retrieved_chunks"]),
            )
        return {"reranked_children": []}

    headers = {
        "Authorization": f"Bearer {os.environ.get('JINA_API_KEY', '')}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": rerank_model,
        "query": state["user_message"],
        "documents": [c["text"] for c in chunks],
        "top_n": top_k,
        # We map results back to chunks by index, so we don't need the text echoed back.
        "return_documents": False,
    }

    result = await _rerank_with_retry(payload, headers)

    reranked = []
    for r in result.get("results", []):
        chunk = dict(chunks[r["index"]])
        # Jina returns 0..1 relevance scores under "relevance_score" (older payloads "score").
        chunk["score"] = r.get("relevance_score", r.get("score", 0.0))
        reranked.append(chunk)

    # Sort by score descending, take top_k
    reranked.sort(key=lambda x: x["score"], reverse=True)
    return {"reranked_children": reranked[:top_k]}
