import asyncio
import logging
import os
import re
import xml.etree.ElementTree as ET

import httpx

from backend.graph.ingestion_graph import run_ingestion
from backend.graph.nodes._llm import groq_complete
from backend.graph.stream import emit

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = "http://www.w3.org/2005/Atom"

# Detect when the user wants recent work, so we sort arXiv by submission date
# instead of relevance.
_RECENCY_RE = re.compile(
    r"\b("
    r"recent|recently|latest|newest|this year|up[\s-]?to[\s-]?date|"
    r"state[\s-]?of[\s-]?the[\s-]?art|sota|2024|2025|2026"
    r")\b",
    re.IGNORECASE,
)


async def _search_arxiv(
    http_client: httpx.AsyncClient,
    query: str,
    max_results: int = 5,
    sort_by_date: bool = False,
) -> list[dict]:
    params = {"search_query": f"all:{query}", "max_results": max_results}
    if sort_by_date:
        params["sortBy"] = "submittedDate"
        params["sortOrder"] = "descending"
    response = await http_client.get(ARXIV_API_URL, params=params)
    response.raise_for_status()
    xml_text = response.text
    papers: list[dict] = []
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
            papers.append({"arxiv_id": arxiv_id, "title": title, "abstract": abstract})
    except ET.ParseError:
        logger.warning("Failed to parse arxiv XML response for query: %r", query)
    return papers


async def ask_corrective_node(state: dict) -> dict:
    query_count = int(os.environ.get("ASK_ARXIV_QUERY_COUNT", "3"))
    ingest_candidates = int(os.environ.get("ASK_INGEST_CANDIDATES", "3"))
    grading_model = os.environ.get("GRADING_MODEL", "llama-3.1-8b-instant")

    user_message = state.get("user_message", "")
    retry_count = state.get("retry_count", 0)
    recency = bool(_RECENCY_RE.search(user_message))

    emit({"type": "status", "step": "searching_arxiv", "content": "Looking up papers on arXiv…"})

    # Generate diverse queries. If the LLM is unavailable, fall back to a plain
    # keyword query from the user's message so the corrective branch still runs.
    try:
        content = await groq_complete(
            model=grading_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Generate exactly {query_count} diverse arxiv search queries to find papers"
                        " that answer the user's question."
                        " Return one query per line, no numbering or bullets."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            max_tokens=80,
        )
        raw_queries = content.strip().splitlines()
        queries = [q.strip() for q in raw_queries if q.strip()][:query_count]
    except Exception as exc:
        logger.warning("arxiv query generation failed (%s); falling back to raw message", exc)
        queries = []
    if not queries:
        queries = [user_message[:200]]

    # Search concurrently. return_exceptions so one failed query doesn't abort the rest.
    all_papers: list[dict] = []
    async with httpx.AsyncClient() as http:
        results = await asyncio.gather(
            *[_search_arxiv(http, q, sort_by_date=recency) for q in queries],
            return_exceptions=True,
        )
    for result in results:
        if isinstance(result, Exception):
            logger.warning("arxiv search failed for a query: %s", result)
            continue
        all_papers.extend(result)

    # Deduplicate by arxiv_id
    seen_ids: set[str] = set()
    unique_papers: list[dict] = []
    for p in all_papers:
        if p["arxiv_id"] not in seen_ids:
            seen_ids.add(p["arxiv_id"])
            unique_papers.append(p)

    # Relevance check: ask grading model to filter candidates
    pool = unique_papers[: ingest_candidates * 2]
    relevant_ids: set[str] = set()
    if pool:
        candidates_text = "\n".join(
            f"ID: {p['arxiv_id']}\nTitle: {p['title']}\nAbstract: {p['abstract'][:300]}"
            for p in pool
        )
        raw_ids = (
            await groq_complete(
                model=grading_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a relevance filter. Given a user question and a list of papers "
                            "(each with an ID, title, and abstract), return a comma-separated list of "
                            "the arxiv IDs that are relevant to the question. "
                            "Return only the IDs, nothing else. If none are relevant, return an empty string."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Question: {user_message}\n\nPapers:\n{candidates_text}",
                    },
                ],
                max_tokens=60,
            )
        ).strip()
        relevant_ids = {rid.strip() for rid in raw_ids.split(",") if rid.strip()}

    # If the relevance check returned nothing parseable, fall back to just the single
    # top-ranked candidate rather than ingesting the whole unfiltered pool — that would
    # bloat the generation context (worsening 413s) with likely-irrelevant papers.
    if not relevant_ids and pool:
        relevant_ids = {pool[0]["arxiv_id"]}

    # Ingest relevant papers concurrently (bounded). Each ingestion is dominated by
    # network I/O (arxiv fetch + Qdrant upsert); the embedding calls inside share a
    # module-level semaphore in embedder.py, so fanning out a couple of ingestions at
    # once is safe and turns 30-90s of sequential work into roughly one paper's time.
    # ASK_INGEST_CONCURRENCY caps how many run at once. Log failures but do not abort.
    to_ingest = [p for p in pool if p["arxiv_id"] in relevant_ids][:ingest_candidates]
    ingest_concurrency = int(os.environ.get("ASK_INGEST_CONCURRENCY", "2"))
    ingest_sem = asyncio.Semaphore(ingest_concurrency)

    async def _safe_ingest(arxiv_id: str) -> None:
        async with ingest_sem:
            try:
                await run_ingestion(arxiv_id)
            except Exception as exc:
                logger.warning("Failed to ingest paper %r: %s", arxiv_id, exc)

    await asyncio.gather(*[_safe_ingest(p["arxiv_id"]) for p in to_ingest])

    new_retry_count = retry_count + 1

    return {
        "arxiv_queries": queries,
        "retry_count": new_retry_count,
    }
