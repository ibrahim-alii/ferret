import logging
import os
import xml.etree.ElementTree as ET

import httpx

from backend.graph.nodes._llm import groq_complete
from backend.graph.stream import emit

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = "http://www.w3.org/2005/Atom"


async def deep_dive_insufficient_node(state: dict) -> dict:
    emit({"type": "status", "step": "searching_arxiv", "content": "Searching arXiv"})
    grading_model = os.environ.get("GRADING_MODEL", "llama-3.1-8b-instant")
    user_message = state.get("user_message", "")

    # Generate arxiv search query using grading model (NOT generation model)
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

    # Search arxiv. A failure here shouldn't error the whole turn — we can still tell
    # the user we lack grounding, just without paper suggestions.
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

    # Parse arxiv XML
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
            abstract_snippet = abstract[:200]

            emit(
                {
                    "type": "citation",
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract_snippet": abstract_snippet,
                }
            )
            citations.append({"arxiv_id": arxiv_id, "title": title})
    except ET.ParseError:
        logger.warning("Failed to parse arxiv XML response for query: %r", query)

    # Emit "not enough info" message token
    not_enough_msg = (
        "I don't have enough information in the current knowledge base to fully answer your question. "
        "Here are some relevant papers I found that may help:"
    )
    emit({"type": "token", "content": not_enough_msg})
    emit({"type": "done"})

    return {"citations": citations}
