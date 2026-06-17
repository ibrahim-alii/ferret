"""Async Qdrant client wrapper — reads connection settings from env."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from qdrant_client import AsyncQdrantClient


@asynccontextmanager
async def get_client() -> AsyncIterator[AsyncQdrantClient]:
    url = os.environ.get("QDRANT_URL")
    if not url:
        raise EnvironmentError("QDRANT_URL environment variable is not set")
    # Coerce empty string to None so the client treats it as unauthenticated
    api_key = os.environ.get("QDRANT_API_KEY") or None
    client = AsyncQdrantClient(url=url, api_key=api_key)
    try:
        yield client
    finally:
        await client.close()
