"""
Section filter: drops low-signal sections before chunking.
Drop list is configurable; Abstract is always kept.
"""
from __future__ import annotations

import logging
import os
import re

from backend.ingestion.blocks import ParsedBlock

log = logging.getLogger(__name__)

_DEFAULT_DROP_PREFIXES: list[str] = [
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "funding",
    "appendix",
    "author contributions",
    "conflict of interest",
    "conflicts of interest",
    "data availability",
    "supplementary",
]

_ALWAYS_KEEP_PREFIXES: list[str] = ["abstract"]

# Sections with fewer prose characters than this (after stripping LaTeX) are
# treated as equation-only and dropped. Configurable via EQUATION_MIN_PROSE_CHARS.
_LATEX_PATTERN = re.compile(
    r"\$\$[\s\S]*?\$\$"       # display math $$...$$
    r"|\$[^$\n]*?\$"           # inline math $...$
    r"|\\[a-zA-Z]+\{[^}]*\}"  # \cmd{...}
    r"|\\[a-zA-Z]+"            # bare \cmd
)


def _is_equation_only(text: str, min_prose_chars: int) -> bool:
    """
    Return True when a section is predominantly LaTeX with little readable prose.
    Only fires when the text actually contains LaTeX-like markup — plain short
    sections (e.g. single-sentence conclusions) are not affected.
    """
    if not _LATEX_PATTERN.search(text):
        return False
    prose = _LATEX_PATTERN.sub("", text).strip()
    return len(prose) < min_prose_chars


def _load_env_drop_list() -> list[str] | None:
    raw = os.environ.get("SECTION_DROP_LIST", "")
    if not raw.strip():
        return None
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def filter_sections(
    blocks: list[ParsedBlock],
    drop_list: list[str] | None = None,
    equation_min_prose_chars: int | None = None,
) -> list[ParsedBlock]:
    """
    Remove blocks that add retrieval noise:
    1. Name-prefix match against drop_list (case-insensitive).
    2. Equation-only heuristic: prose < equation_min_prose_chars after stripping LaTeX.
       Applied to prose blocks only — tables/figures carry little prose by design and
       must never be dropped by this rule.
    Blocks whose section names start with an always-keep prefix are never dropped.
    """
    if drop_list is None:
        drop_list = _load_env_drop_list() or _DEFAULT_DROP_PREFIXES
    if equation_min_prose_chars is None:
        equation_min_prose_chars = int(
            os.environ.get("EQUATION_MIN_PROSE_CHARS", "150")
        )

    drop_lower = [d.lower() for d in drop_list]
    keep_lower = [k.lower() for k in _ALWAYS_KEEP_PREFIXES]

    result: list[ParsedBlock] = []
    for block in blocks:
        name_lower = block.section_name.lower().strip()

        if any(name_lower.startswith(k) for k in keep_lower):
            result.append(block)
            continue

        if any(name_lower.startswith(d) for d in drop_lower):
            log.debug("Dropping section (name match): %r", block.section_name)
            continue

        if block.content_type == "text" and _is_equation_only(
            block.text, equation_min_prose_chars
        ):
            log.debug("Dropping equation-only section: %r", block.section_name)
            continue

        result.append(block)

    return result
