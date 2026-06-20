"""Re-exports for Module 2's Qdrant upsert interface.

Module 2's real implementations live in store.py and models.py; this module
re-exports them so callers can import from a single stable path.
"""
from backend.vectorstore.models import ChunkVector
from backend.vectorstore.store import delete_paper_points, upsert_chunks

__all__ = ["ChunkVector", "delete_paper_points", "upsert_chunks"]
