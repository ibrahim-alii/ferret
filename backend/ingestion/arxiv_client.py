"""
Async arxiv API client.
Sends ARXIV_USER_AGENT header on all requests (arxiv API etiquette).
"""
from __future__ import annotations

import asyncio
import logging
import os
from xml.etree import ElementTree as ET

import httpx
from typing import TypedDict

log = logging.getLogger(__name__)


class ArxivMetadata(TypedDict):
    title: str
    abstract: str
    authors: list[str]
    published_date: str

ARXIV_ATOM_NS = "http://www.w3.org/2005/Atom"
EXPORT_URL = "https://export.arxiv.org/api/query"
HTML_URL = "https://arxiv.org/html/{arxiv_id}"
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"

_USER_AGENT = os.environ.get(
    "ARXIV_USER_AGENT",
    "Ferret/0.1 (research tool; mailto:user@example.com)",
)

_MAX_RETRIES = 4


class ArxivNotFoundError(Exception):
    pass


async def _get_with_retry(
    client: httpx.AsyncClient, url: str, *, params: dict | None = None
) -> httpx.Response:
    """GET with backoff on HTTP 429 (honoring Retry-After) and transient transport errors.

    arxiv's export API rate-limits aggressively; without this a burst of ingests
    fails hard with 429/timeout instead of backing off and succeeding.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = await client.get(url, params=params)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = 2 ** (attempt + 1)
            log.warning("arxiv request to %s failed (%s); retrying in %ds", url, exc, wait)
            await asyncio.sleep(wait)
            continue

        if response.status_code == 429 and attempt < _MAX_RETRIES - 1:
            retry_after = (response.headers.get("Retry-After") or "").strip()
            wait = float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
            log.warning("arxiv 429 for %s; retrying in %.0fs", url, wait)
            await asyncio.sleep(wait)
            continue

        return response

    raise last_exc if last_exc else RuntimeError("unreachable")


async def fetch_metadata(arxiv_id: str) -> ArxivMetadata:
    """Return title, abstract, authors list, and published_date for an arxiv paper."""
    params = {"id_list": arxiv_id, "max_results": "1"}
    async with httpx.AsyncClient(headers={"User-Agent": _USER_AGENT}, timeout=30) as client:
        response = await _get_with_retry(client, EXPORT_URL, params=params)
        response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"atom": ARXIV_ATOM_NS}
    entries = root.findall("atom:entry", ns)
    if not entries:
        raise ArxivNotFoundError(f"No paper found for arxiv_id={arxiv_id!r}")

    entry = entries[0]
    title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
    abstract = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
    published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
    authors = [
        a.findtext("atom:name", default="", namespaces=ns) or ""
        for a in entry.findall("atom:author", ns)
    ]

    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "published_date": published,
    }


async def fetch_html(arxiv_id: str) -> str | None:
    """Fetch arxiv HTML rendering. Returns None if unavailable (404)."""
    url = HTML_URL.format(arxiv_id=arxiv_id)
    async with httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT}, timeout=60, follow_redirects=True
    ) as client:
        response = await _get_with_retry(client, url)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.text


async def download_pdf(arxiv_id: str) -> bytes:
    """Download the PDF bytes for an arxiv paper."""
    url = PDF_URL.format(arxiv_id=arxiv_id)
    async with httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT}, timeout=120, follow_redirects=True
    ) as client:
        response = await _get_with_retry(client, url)
    response.raise_for_status()
    return response.content
