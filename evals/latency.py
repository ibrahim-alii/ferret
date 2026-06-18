"""Async latency tracker for per-stage wall-clock timing and API call counting."""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncIterator, Any


class LatencyTracker:
    def __init__(self) -> None:
        self._stages: dict[str, float] = {}
        self._calls: dict[str, int] = defaultdict(int)

    @asynccontextmanager
    async def track_stage(self, name: str) -> AsyncIterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._stages[name] = self._stages.get(name, 0.0) + elapsed

    def count_call(self, provider: str) -> None:
        self._calls[provider] += 1

    def report(self) -> dict[str, Any]:
        return {
            "stages": dict(self._stages),
            "calls": dict(self._calls),
        }
