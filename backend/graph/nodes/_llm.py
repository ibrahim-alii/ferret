"""Shared Groq helpers for graph nodes: bounded timeout + retry on transient errors.

A new client is constructed per call (cheap) rather than cached, so tests that patch
``groq.AsyncGroq`` keep working and so the timeout can be re-read from env.
"""
from __future__ import annotations

import asyncio
import logging
import os

import groq

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
# Transient failures worth retrying; 4xx request errors are not retried.
_TRANSIENT = (
    groq.RateLimitError,
    groq.APITimeoutError,
    groq.APIConnectionError,
    groq.InternalServerError,
)


def get_groq_client() -> groq.AsyncGroq:
    """Groq client with an explicit timeout so a hung request can't stall the stream."""
    return groq.AsyncGroq(timeout=float(os.environ.get("GROQ_TIMEOUT", "60")))


async def groq_complete(*, model: str, messages: list[dict], max_tokens: int) -> str:
    """Non-streaming chat completion with retry/backoff on transient Groq errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            client = get_groq_client()
            completion = await client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens
            )
            return completion.choices[0].message.content or ""
        except _TRANSIENT as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = 2**attempt
            logger.warning("Groq call failed (%s); retrying in %ds", exc, wait)
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")
