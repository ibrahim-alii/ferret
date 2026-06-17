"""
Async arxiv API client.
Sends ARXIV_USER_AGENT header on all requests (arxiv API etiquette).
"""
from __future__ import annotations

import os
from xml.etree import ElementTree as ET

import httpx
from typing import TypedDict


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


class ArxivNotFoundError(Exception):
    pass


async def fetch_metadata(arxiv_id: str) -> ArxivMetadata:
    """Return title, abstract, authors list, and published_date for an arxiv paper."""
    params = {"id_list": arxiv_id, "max_results": "1"}
    async with httpx.AsyncClient(headers={"User-Agent": _USER_AGENT}, timeout=30) as client:
        response = await client.get(EXPORT_URL, params=params)
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
        response = await client.get(url)
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
        response = await client.get(url)
    response.raise_for_status()
    return response.content
