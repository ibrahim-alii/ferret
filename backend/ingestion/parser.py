"""
Paper parser: HTML-first (arxiv.org/html), PDF fallback (PyMuPDF).
Both paths return an ordered list of ParsedBlock (prose / table markdown / figure caption).

Multimodal extraction (no VLM — arXiv tables/figures always carry captions):
- Tables  -> serialized to GitHub-flavored markdown (HTML <table>; PDF find_tables()).
- Figures -> caption text + an image reference. HTML hotlinks the absolute arxiv.org
  <img> URL; PDF-only papers have their image bytes extracted to MEDIA_DIR and served
  at /media/<arxiv_id>/<file>.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from urllib.parse import urljoin

from backend.ingestion.blocks import ParsedBlock

log = logging.getLogger(__name__)

_FIGURE_CAPTION_RE = re.compile(r"^\s*(figure|fig\.?)\s*\d+", re.IGNORECASE)


def _media_dir() -> Path:
    return Path(os.environ.get("MEDIA_DIR", "./backend/data/media"))


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #


def _html_table_to_markdown(table) -> str:
    """Serialize a BeautifulSoup <table> to GitHub-flavored markdown.

    The first row is treated as the header. Pipes inside cells are escaped and
    whitespace collapsed so the grid stays on single lines.
    """
    rows = table.find_all("tr")
    if not rows:
        return ""

    def cells(tr) -> list[str]:
        out = []
        for cell in tr.find_all(["th", "td"]):
            text = cell.get_text(separator=" ", strip=True)
            out.append(re.sub(r"\s+", " ", text).replace("|", r"\|"))
        return out

    header = cells(rows[0])
    if not header:
        return ""
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for tr in rows[1:]:
        body = cells(tr)
        if not body:
            continue
        # Pad/trim to the header width so the markdown stays well-formed.
        body = (body + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(body) + " |")
    return "\n".join(lines)


def _parse_html(html_content: str, arxiv_id: str) -> list[ParsedBlock]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "lxml")
    base = f"https://arxiv.org/html/{arxiv_id}/"
    blocks: list[ParsedBlock] = []

    for sec in soup.find_all("section"):
        heading = sec.find(re.compile(r"^h[1-6]$"))
        if not heading:
            continue
        section_name = heading.get_text(separator=" ", strip=True)
        heading.decompose()

        # Pull tables/figures out FIRST so their text doesn't leak into the prose
        # block (and so a table isn't flattened into a wall of cell text).
        for fig in sec.find_all("figure"):
            caption_el = fig.find("figcaption")
            caption = caption_el.get_text(separator=" ", strip=True) if caption_el else ""
            inner_table = fig.find("table")
            if inner_table is not None:
                md = _html_table_to_markdown(inner_table)
                if md:
                    text = f"{caption}\n\n{md}" if caption else md
                    blocks.append(ParsedBlock(section_name, text, "table"))
            else:
                img = fig.find("img")
                src = img.get("src") if img else None
                url = urljoin(base, src) if src else None
                if url and not url.startswith(("http://", "https://")):
                    url = None
                if caption or url:
                    blocks.append(
                        ParsedBlock(section_name, caption, "figure", media_url=url)
                    )
            fig.decompose()

        # Standalone tables not wrapped in a <figure>.
        for table in sec.find_all("table"):
            md = _html_table_to_markdown(table)
            if md:
                blocks.append(ParsedBlock(section_name, md, "table"))
            table.decompose()

        body = sec.get_text(separator=" ", strip=True)
        if body:
            blocks.append(ParsedBlock(section_name, body, "text"))

    return blocks


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


def _extract_pdf_image(doc, page, caption_bbox, arxiv_id: str) -> str | None:
    """Extract the image nearest a figure caption on this page and save it.

    Returns a "/media/<arxiv_id>/<file>" url, or None if no image is found.
    Heuristic: the figure image is the one whose bbox is closest (vertically) to
    the caption, which on arXiv PDFs sits just below the figure.
    """
    import fitz

    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        return None
    candidates = [i for i in infos if i.get("xref")]
    if not candidates:
        return None

    cap = fitz.Rect(caption_bbox)
    cap_top = cap.y0

    def gap(info) -> float:
        bbox = fitz.Rect(info["bbox"])
        # Prefer images sitting above the caption; penalise distance.
        return abs(cap_top - bbox.y1)

    best = min(candidates, key=gap)
    xref = best["xref"]
    try:
        extracted = doc.extract_image(xref)
    except Exception:
        return None
    if not extracted or not extracted.get("image"):
        return None

    ext = extracted.get("ext", "png")
    out_dir = _media_dir() / arxiv_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"p{page.number}-{xref}.{ext}"
    (out_dir / fname).write_bytes(extracted["image"])
    return f"/media/{arxiv_id}/{fname}"


def _parse_pdf(pdf_bytes: bytes, arxiv_id: str) -> list[ParsedBlock]:
    import fitz  # PyMuPDF

    blocks: list[ParsedBlock] = []
    current_section: str | None = None
    current_lines: list[str] = []

    def flush_text() -> None:
        nonlocal current_lines
        if current_section is not None and current_lines:
            blocks.append(ParsedBlock(current_section, " ".join(current_lines), "text"))
        current_lines = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            # Tables first; record their bboxes so we can skip those text lines.
            table_rects: list[fitz.Rect] = []
            try:
                found = page.find_tables()
                for table in found.tables:
                    md = (table.to_markdown() or "").strip()
                    if md:
                        section = current_section or "Table"
                        blocks.append(ParsedBlock(section, md, "table"))
                        table_rects.append(fitz.Rect(table.bbox))
            except Exception as exc:
                log.debug("PDF table extraction failed on page %s: %s", page.number, exc)

            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    line_text = " ".join(s["text"] for s in spans).strip()
                    if not line_text:
                        continue

                    line_rect = fitz.Rect(line["bbox"])
                    if any(r.intersects(line_rect) for r in table_rects):
                        continue  # already captured as a markdown table

                    if _FIGURE_CAPTION_RE.match(line_text):
                        url = _extract_pdf_image(doc, page, line["bbox"], arxiv_id)
                        section = current_section or "Figure"
                        blocks.append(
                            ParsedBlock(section, line_text, "figure", media_url=url)
                        )
                        continue

                    max_size = max(s["size"] for s in spans)
                    is_bold = any(s["flags"] & 2**4 for s in spans)
                    if (max_size >= 13 or is_bold) and len(line_text) < 120:
                        flush_text()
                        current_section = line_text
                    else:
                        current_lines.append(line_text)

    if current_section is not None and current_lines:
        blocks.append(ParsedBlock(current_section, " ".join(current_lines), "text"))
    elif current_lines and not blocks:
        blocks.append(ParsedBlock("Body", " ".join(current_lines), "text"))

    return blocks


def parse(
    html_content: str | None,
    pdf_bytes: bytes | None,
    arxiv_id: str,
) -> list[ParsedBlock]:
    """
    Parse a paper into ordered ParsedBlocks (prose / table / figure).
    Prefers HTML; falls back to PDF; returns [] if both fail (abstract_only degradation).
    arxiv_id is used to absolutize HTML image URLs and to name extracted PDF images.
    """
    if html_content:
        try:
            blocks = _parse_html(html_content, arxiv_id)
            if blocks:
                log.debug("Parsed %d blocks from HTML", len(blocks))
                return blocks
        except Exception as exc:
            log.warning("HTML parse failed: %s", exc, exc_info=True)

    if pdf_bytes:
        try:
            blocks = _parse_pdf(pdf_bytes, arxiv_id)
            if blocks:
                log.debug("Parsed %d blocks from PDF", len(blocks))
                return blocks
        except Exception as exc:
            log.warning("PDF parse failed: %s", exc, exc_info=True)

    log.info("Both HTML and PDF parsing failed; degrading to abstract_only")
    return []
