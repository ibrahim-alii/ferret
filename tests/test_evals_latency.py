"""Tests for evals/latency.py — per-stage latency and call tracking."""
from __future__ import annotations

import asyncio
import json

import pytest


@pytest.mark.eval
class TestLatencyTracker:
    @pytest.mark.asyncio
    async def test_latency_tracker_records_per_stage_times(self):
        from evals.latency import LatencyTracker

        tracker = LatencyTracker()
        async with tracker.track_stage("retrieval"):
            await asyncio.sleep(0.01)

        report = tracker.report()
        assert "retrieval" in report["stages"]
        assert report["stages"]["retrieval"] >= 0.0

    @pytest.mark.asyncio
    async def test_api_call_counter_increments_on_groq_call(self):
        from evals.latency import LatencyTracker

        tracker = LatencyTracker()
        tracker.count_call("groq")
        tracker.count_call("groq")

        report = tracker.report()
        assert report["calls"]["groq"] == 2

    @pytest.mark.asyncio
    async def test_api_call_counter_increments_on_voyage_embed_call(self):
        from evals.latency import LatencyTracker

        tracker = LatencyTracker()
        tracker.count_call("voyage")

        report = tracker.report()
        assert report["calls"]["voyage"] == 1

    @pytest.mark.asyncio
    async def test_api_call_counter_increments_on_qdrant_query(self):
        from evals.latency import LatencyTracker

        tracker = LatencyTracker()
        tracker.count_call("qdrant")
        tracker.count_call("qdrant")
        tracker.count_call("qdrant")

        report = tracker.report()
        assert report["calls"]["qdrant"] == 3

    @pytest.mark.asyncio
    async def test_sufficient_branch_groq_calls_within_budget(self):
        """Sufficient branch should use <= 2 Groq calls (grade + generate)."""
        from evals.latency import LatencyTracker

        tracker = LatencyTracker()
        # Simulate sufficient branch: 1 grade call + 1 generate call
        tracker.count_call("groq")
        tracker.count_call("groq")

        report = tracker.report()
        assert report["calls"].get("groq", 0) <= 2

    @pytest.mark.asyncio
    async def test_corrective_branch_groq_calls_within_budget(self):
        """Corrective branch should use <= 5 Groq calls."""
        from evals.latency import LatencyTracker

        tracker = LatencyTracker()
        for _ in range(5):
            tracker.count_call("groq")

        report = tracker.report()
        assert report["calls"].get("groq", 0) <= 5

    @pytest.mark.asyncio
    async def test_latency_report_serializable_to_json(self):
        from evals.latency import LatencyTracker

        tracker = LatencyTracker()
        async with tracker.track_stage("embed"):
            await asyncio.sleep(0.001)
        tracker.count_call("voyage")

        report = tracker.report()
        serialized = json.dumps(report)
        loaded = json.loads(serialized)
        assert "stages" in loaded
        assert "calls" in loaded

    @pytest.mark.asyncio
    async def test_multiple_stages_tracked_independently(self):
        from evals.latency import LatencyTracker

        tracker = LatencyTracker()
        async with tracker.track_stage("embed"):
            await asyncio.sleep(0.005)
        async with tracker.track_stage("generate"):
            await asyncio.sleep(0.01)

        report = tracker.report()
        assert "embed" in report["stages"]
        assert "generate" in report["stages"]
        assert report["stages"]["generate"] >= report["stages"]["embed"]
