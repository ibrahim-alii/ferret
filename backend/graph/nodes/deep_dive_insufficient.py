import logging
import os
import xml.etree.ElementTree as ET

import aiosqlite
import httpx

from backend.graph.nodes._llm import groq_complete
from backend.graph.stream import emit
from backend.ingestion.embedder import embed_query
from backend.vectorstore.store import hybrid_search

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = "http://www.w3.org/2005/Atom"

NOT_ENOUGH_MSG = (
    "I don't have enough information in the current knowledge base to fully answer your question. "
    "Here are some relevant papers I found that may help:"
)


async def _suggest_from_corpus(state: dict) -> list[dict]:
    """Cross-corpus retrieval to surface related papers the user has already ingested.

    Deep Dive is single-paper, strictly grounded: we never *answer* from other papers.
    But when the chosen paper falls short, the most relevant suggestions are the ones
    already in the corpus, ranked by relevance to the question — better than a generic
    arXiv keyword search. We retrieve cross-corpus (paper_id=None), drop the current
    paper, and surface the distinct papers as suggestions only (no chunk text feeds
    generation). Reuses the cached query vector so there's no extra embedding call.
    """
    current_paper = state.get("paper_id")
    limit = int(os.environ.get("RERANK_CANDIDATE_COUNT", "30"))
    suggest_count = int(os.environ.get("DEEP_DIVE_SUGGEST_COUNT", "5"))

    query_dense = state.get("query_vector") or await embed_query(state["user_message"])
    chunks = await hybrid_search(
        query_dense=query_dense,
        query_text=state["user_message"],
        paper_id=None,
        limit=limit,
    )

    # Distinct papers, best-rank-first, excluding the paper being deep-dived.
    ordered_ids: list[str] = []
    for c in chunks:
        pid = c.paper_id
        if not pid or pid == current_paper or pid in ordered_ids:
            continue
        ordered_ids.append(pid)
        if len(ordered_ids) >= suggest_count:
            break

    if not ordered_ids:
        return []

    # Titles/abstracts live in SQLite (papers), not Qdrant; one read-only lookup.
    meta: dict[str, tuple[str, str]] = {}
    db_path = os.environ.get("SQLITE_DB_PATH", "./backend/data/app.db")
    placeholders = ",".join("?" * len(ordered_ids))
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(
            f"SELECT arxiv_id, title, abstract FROM papers WHERE arxiv_id IN ({placeholders})",
            ordered_ids,
        )
        async for row in cursor:
            meta[row["arxiv_id"]] = (row["title"] or "", row["abstract"] or "")

    citations: list[dict] = []
    for arxiv_id in ordered_ids:
        title, abstract = meta.get(arxiv_id, ("", ""))
        emit(
            {
                "type": "citation",
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract_snippet": abstract[:200] if abstract else None,
            }
        )
        citations.append({"arxiv_id": arxiv_id, "title": title})
    return citations


async def _suggest_from_arxiv(state: dict) -> list[dict]:
    """Fallback when the corpus has no related papers (e.g. only one paper ingested).

    Searches arXiv for candidates the user could ingest. A failure here shouldn't error
    the whole turn — we can still tell the user we lack grounding, just without suggestions.
    """
    grading_model = os.environ.get("GRADING_MODEL", "llama-3.1-8b-instant")
    user_message = state.get("user_message", "")

    query = (
        await groq_complete(
            model=grading_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a concise arxiv search query for the following question. "
                        "Return only the query string, nothing else."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            max_tokens=100,
        )
    ).strip()

    citations: list[dict] = []
    xml_text = ""
    timeout = float(os.environ.get("ARXIV_TIMEOUT", "30"))
    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            response = await http.get(
                ARXIV_API_URL,
                params={"search_query": f"all:{query}", "max_results": 5},
            )
            response.raise_for_status()
            xml_text = response.text
    except httpx.HTTPError as exc:
        logger.warning("arxiv search failed in deep-dive fallback: %s", exc)

    try:
        root = ET.fromstring(xml_text)
        entries = root.findall(f"{{{ARXIV_NS}}}entry")
        for entry in entries:
            id_elem = entry.find(f"{{{ARXIV_NS}}}id")
            title_elem = entry.find(f"{{{ARXIV_NS}}}title")
            summary_elem = entry.find(f"{{{ARXIV_NS}}}summary")

            if id_elem is None:
                continue

            raw_id = id_elem.text.strip()
            arxiv_id = raw_id.split("/abs/")[-1].split("v")[0] if "/abs/" in raw_id else raw_id
            title = title_elem.text.strip() if title_elem is not None else ""
            abstract = summary_elem.text.strip() if summary_elem is not None else ""

            emit(
                {
                    "type": "citation",
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract_snippet": abstract[:200],
                }
            )
            citations.append({"arxiv_id": arxiv_id, "title": title})
    except ET.ParseError:
        logger.warning("Failed to parse arxiv XML response for query: %r", query)

    return citations


async def deep_dive_insufficient_node(state: dict) -> dict:
    # Drift guard: a Deep Dive that retrieved *nothing* means the chosen paper has no
    # vectors in the index (SQLite<->Qdrant drift), not that the question is off-topic.
    # Suggesting arbitrary papers here is misleading, so return an honest message instead.
    if state.get("mode") == "deep_dive" and not state.get("retrieved_chunks"):
        emit(
            {
                "type": "token",
                "content": (
                    "I couldn't find this content in the selected paper — its search "
                    "index may be out of sync. Try re-ingesting the paper and asking again."
                ),
            }
        )
        emit({"type": "done"})
        return {"citations": []}

    emit({"type": "status", "step": "searching_corpus", "content": "Looking up related papers…"})

    # Corpus-first: suggest already-ingested papers ranked by relevance. Fall back to an
    # arXiv search only when the corpus has nothing else to offer.
    citations = await _suggest_from_corpus(state)
    if not citations:
        citations = await _suggest_from_arxiv(state)

    emit({"type": "token", "content": NOT_ENOUGH_MSG})
    emit({"type": "done"})

    return {"citations": citations}
