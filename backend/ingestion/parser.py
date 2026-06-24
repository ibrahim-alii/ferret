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
# arXiv ids may contain a slash (old style, e.g. "hep-th/9901001"); allow that but
# strip anything that could escape the media directory.
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._/-]")
# arXiv MathML emits zero-width chars between sub/superscript glyphs (e.g.
# "P_drop" renders as "P d​r​o​p"). They survive get_text() and
# corrupt table cells and prose, so strip them from the HTML before parsing.
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")
# Image extensions PyMuPDF may report; anything else is coerced to png so a crafted
# PDF can't smuggle path separators through extract_image()'s ext field.
_ALLOWED_IMG_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}
# PDF figure capture: render scale (~216 DPI, crisp vector labels), how far above
# the caption a figure may extend, and how far above the rasters to pull in labels.
_FIGURE_RENDER_ZOOM = 3.0
_MAX_FIGURE_HEIGHT = 460.0
_FIGURE_LABEL_PAD = 20.0
# Caption continuation: lines within this vertical gap (points) of the previous
# caption line belong to the same caption; a larger gap ends it.
_CAPTION_LINE_GAP = 14.0
_CAPTION_MAX_LINES = 8


def _safe_id(arxiv_id: str) -> str:
    # Strip leading/trailing slashes too: a leading slash makes `_media_dir() / id`
    # an absolute path that escapes MEDIA_DIR (pathlib resets on an absolute rhs).
    return _SAFE_ID_RE.sub("", arxiv_id).replace("..", "").strip("/")


def _media_dir() -> Path:
    # Resolve to an absolute path so the parser (run via asyncio.to_thread) and the
    # FastAPI StaticFiles mount always agree regardless of the process CWD.
    return Path(os.environ.get("MEDIA_DIR", "./backend/data/media")).resolve()


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

    # Strip zero-width chars up front so they can't survive get_text() into any
    # extracted block (table cells, captions, prose). See _ZERO_WIDTH_RE.
    html_content = _ZERO_WIDTH_RE.sub("", html_content)
    soup = BeautifulSoup(html_content, "lxml")
    # arXiv HTML renders math as MathML that carries the raw TeX in an
    # <annotation encoding="application/x-tex"> sibling of the rendered glyph.
    # get_text() would otherwise emit both (e.g. "↓ \downarrow") and leak the
    # macro into prose and table cells, so drop the annotation source up front.
    for ann in soup.find_all("annotation"):
        ann.decompose()
    # arXiv serves HTML at /html/<id>v<n> and its <img> srcs are paths relative
    # to /html/ that already include the versioned dir (e.g. "<id>v7/Figures/x.png").
    # So the base is /html/ itself; injecting the id here would double it
    # ("/html/<id>/<id>v7/...") and 404.
    base = "https://arxiv.org/html/"
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
                # Only hotlink images under this paper's own HTML dir
                # (/html/<id>...); a crafted src must not point the rendered <img>
                # at another paper or off arxiv entirely.
                allowed_prefix = f"https://arxiv.org/html/{_safe_id(arxiv_id)}"
                if url and not url.startswith(allowed_prefix):
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
    """Render the figure region above a caption to an image and save it.

    Returns a "/media/<arxiv_id>/<file>" url, or None if no figure is found.

    arXiv figures are usually a raster on a transparent canvas PLUS vector text
    drawn on the page (column headers, row/axis labels). extract_image() would
    grab only the raster — missing the labels, compositing transparency to black,
    and ignoring sibling raster fragments. So we rasterise the page region instead:
    the union of the image bboxes just above the caption, grown to include the
    surrounding label text, clamped to stop just above the caption. This captures
    the figure exactly as it appears in the paper (on the white page background).
    """
    import fitz

    cap_top = fitz.Rect(caption_bbox).y0

    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        return None
    img_rects = [
        fitz.Rect(i["bbox"])
        for i in infos
        if i.get("bbox") and 0 < (cap_top - fitz.Rect(i["bbox"]).y1) < _MAX_FIGURE_HEIGHT
    ]
    if not img_rects:
        return None

    # Union the raster pieces belonging to this figure.
    region = fitz.Rect(img_rects[0])
    for r in img_rects[1:]:
        region |= r

    # Grow to include label text sitting in the figure band (headers just above,
    # row/axis labels beside the rasters), but never below the caption.
    band_top = region.y0 - _FIGURE_LABEL_PAD
    try:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                lr = fitz.Rect(line["bbox"])
                if lr.y0 >= band_top and lr.y1 <= cap_top - 1:
                    region |= lr
    except Exception:
        pass

    region.y1 = min(region.y1, cap_top - 2)
    region += (-4, -4, 4, 4)  # small padding
    region &= page.rect
    if region.is_empty or region.width < 8 or region.height < 8:
        return None

    try:
        zoom = fitz.Matrix(_FIGURE_RENDER_ZOOM, _FIGURE_RENDER_ZOOM)
        data = page.get_pixmap(clip=region, matrix=zoom).tobytes("png")
    except Exception:
        return None
    if not data:
        return None

    safe_id = _safe_id(arxiv_id)
    media_root = _media_dir()
    out_dir = (media_root / safe_id).resolve()
    # Defense-in-depth: never write outside MEDIA_DIR even if a crafted id slips past
    # the regex/strip above.
    if not out_dir.is_relative_to(media_root):
        log.warning("Rejected media path outside MEDIA_DIR for arxiv_id=%r", arxiv_id)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"p{page.number}-{int(region.x0)}-{int(region.y0)}.png"
    (out_dir / fname).write_bytes(data)
    return f"/media/{safe_id}/{fname}"


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

            # Flatten the page's text lines so a multi-line figure caption can be
            # gathered with a forward look (captions wrap across several lines).
            page_lines: list[tuple[str, dict, list]] = []
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = " ".join(s["text"] for s in spans).strip()
                    if text:
                        page_lines.append((text, line, spans))

            i = 0
            while i < len(page_lines):
                line_text, line, spans = page_lines[i]
                line_rect = fitz.Rect(line["bbox"])
                if any(r.intersects(line_rect) for r in table_rects):
                    i += 1
                    continue  # already captured as a markdown table

                if _FIGURE_CAPTION_RE.match(line_text):
                    # Gather the full caption: subsequent tightly-spaced lines until
                    # a paragraph break or the next figure caption.
                    parts = [line_text]
                    prev = line_rect
                    j = i + 1
                    while j < len(page_lines) and len(parts) < _CAPTION_MAX_LINES:
                        nxt_text, nxt_line, _ = page_lines[j]
                        nxt_rect = fitz.Rect(nxt_line["bbox"])
                        if _FIGURE_CAPTION_RE.match(nxt_text):
                            break
                        if nxt_rect.y0 - prev.y1 > _CAPTION_LINE_GAP:
                            break
                        parts.append(nxt_text)
                        prev = nxt_rect
                        j += 1
                    caption = re.sub(r"\s+", " ", " ".join(parts)).strip()
                    url = _extract_pdf_image(doc, page, line["bbox"], arxiv_id)
                    section = current_section or "Figure"
                    blocks.append(ParsedBlock(section, caption, "figure", media_url=url))
                    i = j
                    continue

                max_size = max(s["size"] for s in spans)
                is_bold = any(s["flags"] & 2**4 for s in spans)
                if (max_size >= 13 or is_bold) and len(line_text) < 120:
                    flush_text()
                    current_section = line_text
                else:
                    current_lines.append(line_text)
                i += 1

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
