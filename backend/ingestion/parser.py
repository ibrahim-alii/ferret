"""
Paper parser: HTML-first (arxiv.org/html), PDF fallback (PyMuPDF).
Both paths return an ordered list of (section_name, text) tuples.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)


def _parse_html(html_content: str) -> list[tuple[str, str]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "lxml")
    sections: list[tuple[str, str]] = []

    for sec in soup.find_all("section"):
        heading = sec.find(re.compile(r"^h[1-6]$"))
        if not heading:
            continue
        section_name = heading.get_text(separator=" ", strip=True)
        heading.decompose()
        body = sec.get_text(separator=" ", strip=True)
        if body:
            sections.append((section_name, body))

    return sections


def _parse_pdf(pdf_bytes: bytes) -> list[tuple[str, str]]:
    import fitz  # PyMuPDF

    sections: list[tuple[str, str]] = []
    current_section: str | None = None
    current_lines: list[str] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    line_text = " ".join(s["text"] for s in spans).strip()
                    if not line_text:
                        continue
                    max_size = max(s["size"] for s in spans)
                    is_bold = any(s["flags"] & 2**4 for s in spans)
                    if (max_size >= 13 or is_bold) and len(line_text) < 120:
                        if current_section is not None and current_lines:
                            sections.append((current_section, " ".join(current_lines)))
                        current_section = line_text
                        current_lines = []
                    else:
                        current_lines.append(line_text)

    if current_section is not None and current_lines:
        sections.append((current_section, " ".join(current_lines)))
    elif current_lines and not sections:
        sections.append(("Body", " ".join(current_lines)))

    return sections


def parse(
    html_content: str | None,
    pdf_bytes: bytes | None,
) -> list[tuple[str, str]]:
    """
    Parse a paper into ordered (section_name, text) pairs.
    Prefers HTML; falls back to PDF; returns [] if both fail (abstract_only degradation).
    """
    if html_content:
        try:
            sections = _parse_html(html_content)
            if sections:
                log.debug("Parsed %d sections from HTML", len(sections))
                return sections
        except Exception as exc:
            log.warning("HTML parse failed: %s", exc, exc_info=True)

    if pdf_bytes:
        try:
            sections = _parse_pdf(pdf_bytes)
            if sections:
                log.debug("Parsed %d sections from PDF", len(sections))
                return sections
        except Exception as exc:
            log.warning("PDF parse failed: %s", exc, exc_info=True)

    log.info("Both HTML and PDF parsing failed; degrading to abstract_only")
    return []
