"""Tests for evals/retrieval.py — ranx-based retrieval evaluation."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _make_scored_chunks(chunk_ids: list[str]) -> list[dict]:
    return [
        {"chunk_id": cid, "score": 1.0 - i * 0.1, "text": f"text {cid}"}
        for i, cid in enumerate(chunk_ids)
    ]


@pytest.mark.eval
class TestRetrievalEval:
    def test_ranx_run_built_correctly_from_hybrid_search_results(self):
        from evals.retrieval import build_run

        results = {
            "q1": _make_scored_chunks(["c1", "c2", "c3"]),
            "q2": _make_scored_chunks(["c4", "c5"]),
        }
        run = build_run(results, run_name="hybrid")

        assert run is not None

    def test_recall_at_5_above_minimum_threshold(self):
        from evals.retrieval import compute_metrics

        qrels_dict = {"q1": {"c1": 1, "c2": 1}}
        run_dict = {"q1": {"c1": 1.0, "c2": 0.9, "c3": 0.8, "c4": 0.7, "c5": 0.6}}

        metrics = compute_metrics(qrels_dict, run_dict)

        assert metrics["recall@5"] >= 0.5

    def test_ndcg_at_5_above_minimum_threshold(self):
        from evals.retrieval import compute_metrics

        qrels_dict = {"q1": {"c1": 1}}
        run_dict = {"q1": {"c1": 1.0, "c2": 0.5}}

        metrics = compute_metrics(qrels_dict, run_dict)

        assert metrics["ndcg@5"] >= 0.5

    def test_mrr_above_minimum_threshold(self):
        from evals.retrieval import compute_metrics

        qrels_dict = {"q1": {"c1": 1}}
        run_dict = {"q1": {"c1": 1.0, "c2": 0.5}}

        metrics = compute_metrics(qrels_dict, run_dict)

        assert metrics["mrr"] >= 0.5

    def test_ranx_comparison_report_serializable_to_json(self, tmp_path):
        from evals.retrieval import build_comparison_report

        runs_dict = {
            "dense": {"q1": {"c1": 0.9}},
            "hybrid": {"q1": {"c1": 1.0, "c2": 0.8}},
            "hybrid_rerank": {"q1": {"c1": 1.0, "c2": 0.9}},
        }
        qrels_dict = {"q1": {"c1": 1}}

        report = build_comparison_report(qrels_dict, runs_dict)

        serialized = json.dumps(report)
        loaded = json.loads(serialized)
        assert "runs" in loaded
        assert set(loaded["runs"].keys()) == {"dense", "hybrid", "hybrid_rerank"}

    def test_dense_only_run_lower_ndcg_than_hybrid_run(self):
        """Hypothesis: hybrid >= dense on ndcg@5. Log result — not always guaranteed."""
        from evals.retrieval import compute_metrics

        qrels_dict = {"q1": {"c1": 1, "c2": 1}}
        # Dense-only misses c2; hybrid surfaces it
        dense_run = {"q1": {"c1": 1.0, "c3": 0.5}}
        hybrid_run = {"q1": {"c1": 1.0, "c2": 0.9, "c3": 0.5}}

        dense_metrics = compute_metrics(qrels_dict, dense_run)
        hybrid_metrics = compute_metrics(qrels_dict, hybrid_run)

        # Log result; not a hard assertion since it depends on data
        assert hybrid_metrics["ndcg@5"] >= dense_metrics["ndcg@5"]

    def test_hybrid_run_lower_ndcg_than_hybrid_plus_rerank_run(self):
        """Hypothesis: reranked >= hybrid on ndcg@5 when rerank promotes the right doc."""
        from evals.retrieval import compute_metrics

        qrels_dict = {"q1": {"c1": 1}}
        # Hybrid ranks c2 first (wrong); rerank corrects order to c1 first
        hybrid_run = {"q1": {"c2": 1.0, "c1": 0.5}}
        rerank_run = {"q1": {"c1": 1.0, "c2": 0.5}}

        hybrid_metrics = compute_metrics(qrels_dict, hybrid_run)
        rerank_metrics = compute_metrics(qrels_dict, rerank_run)

        assert rerank_metrics["ndcg@5"] >= hybrid_metrics["ndcg@5"]
