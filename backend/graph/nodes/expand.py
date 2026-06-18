import logging
import os

import aiosqlite

logger = logging.getLogger(__name__)


async def expand_node(state: dict) -> dict:
    max_parents = int(os.environ.get("GENERATION_MAX_PARENTS", "5"))
    db_path = os.environ.get("SQLITE_DB_PATH", "ferret.db")

    children = state["reranked_children"]
    seen_parent_ids: set[str] = set()
    parent_sections: list[dict] = []

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        for child in children:
            if len(parent_sections) >= max_parents:
                break
            parent_id = child.get("parent_chunk_id")
            if parent_id is None or parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_id)

            cursor = await conn.execute(
                "SELECT chunk_id, parent_chunk_id, paper_id, section_title, text"
                " FROM chunks WHERE chunk_id = ?",
                (parent_id,),
            )
            row = await cursor.fetchone()
            if row is not None:
                parent_sections.append(
                    {
                        "section_id": row["chunk_id"],
                        "text": row["text"],
                        "paper_id": row["paper_id"],
                        "section_title": row["section_title"],
                    }
                )

    return {"parent_sections": parent_sections}
