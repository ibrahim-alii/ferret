"""
Section filter: drops low-signal sections before chunking.
Drop list is configurable; Abstract is always kept.
"""
from __future__ import annotations

import logging
import os

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


def _load_env_drop_list() -> list[str] | None:
    raw = os.environ.get("SECTION_DROP_LIST", "")
    if not raw.strip():
        return None
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def filter_sections(
    sections: list[tuple[str, str]],
    drop_list: list[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Remove sections whose names start with a drop-list prefix (case-insensitive).
    Sections whose names start with an always-keep prefix are never dropped.
    """
    if drop_list is None:
        drop_list = _load_env_drop_list() or _DEFAULT_DROP_PREFIXES

    drop_lower = [d.lower() for d in drop_list]
    keep_lower = [k.lower() for k in _ALWAYS_KEEP_PREFIXES]

    result: list[tuple[str, str]] = []
    for name, text in sections:
        name_lower = name.lower().strip()
        if any(name_lower.startswith(k) for k in keep_lower):
            result.append((name, text))
            continue
        if any(name_lower.startswith(d) for d in drop_lower):
            log.debug("Dropping section: %r", name)
            continue
        result.append((name, text))

    return result
