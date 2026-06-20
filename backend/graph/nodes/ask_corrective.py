import asyncio
import logging
import os
import xml.etree.ElementTree as ET

import httpx

from backend.graph.ingestion_graph import run_ingestion
from backend.graph.nodes._llm import groq_complete
from backend.graph.stream import emit

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = "http://www.w3.org/2005/Atom"


async def _search_arxiv(http_client: httpx.AsyncClient, query: str, max_results: int = 5) -> list[dict]:
    response = await http_client.get(
        ARXIV_API_URL,
        params={"search_query": f"all:{query}", "max_results": max_results},
    )
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

    emit({"type": "status", "step": "searching_arxiv", "content": "Searching arXiv for new papers"})
    emit({"type": "interim_message", "content": "Searching for additional sources..."})

    # Generate diverse queries
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
        max_tokens=200,
    )
    raw_queries = content.strip().splitlines()
    queries = [q.strip() for q in raw_queries if q.strip()][:query_count]

    # Search concurrently
    all_papers: list[dict] = []
    async with httpx.AsyncClient() as http:
        results = await asyncio.gather(*[_search_arxiv(http, q) for q in queries])
    for result in results:
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
                max_tokens=200,
            )
        ).strip()
        relevant_ids = {rid.strip() for rid in raw_ids.split(",") if rid.strip()}

    # If relevance check returned nothing useful, fall back to the pool
    if not relevant_ids:
        relevant_ids = {p["arxiv_id"] for p in pool}

    # Ingest only relevant papers up to ingest_candidates. Run sequentially: each
    # ingestion makes batched embedding calls (OpenAI/Gemini), and the user already
    # sees the interim "searching..." message, so fanning these out concurrently would
    # only risk tripping the embedding rate limit. Log failures but do not abort.
    to_ingest = [p for p in pool if p["arxiv_id"] in relevant_ids][:ingest_candidates]
    for paper in to_ingest:
        try:
            await run_ingestion(paper["arxiv_id"])
        except Exception as exc:
            logger.warning("Failed to ingest paper %r: %s", paper["arxiv_id"], exc)

    new_retry_count = retry_count + 1

    return {
        "arxiv_queries": queries,
        "retry_count": new_retry_count,
    }
