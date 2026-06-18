"""Tests for evals/grading.py — grader accuracy evaluation."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.eval
class TestGradingEval:
    @pytest.mark.asyncio
    async def test_grader_false_negative_rate_below_threshold(self):
        """FNR must be below 0.3 — grader should not miss relevant chunks."""
        from evals.grading import compute_grading_metrics

        # Pairs where chunks ARE relevant (score=0.8, above HIGH_THRESHOLD=0.6)
        pairs = [
            {"query": "What is X?", "chunks": [{"score": 0.8, "text": "X is defined as..."}], "label": True},
            {"query": "How does Y work?", "chunks": [{"score": 0.75, "text": "Y works by..."}], "label": True},
            {"query": "What is Z?", "chunks": [{"score": 0.65, "text": "Z is a method..."}], "label": True},
        ]

        with patch("evals.grading._run_grade_node", new_callable=AsyncMock) as mock_grade:
            mock_grade.side_effect = [
                {"grade_result": {"sufficient": True}},
                {"grade_result": {"sufficient": True}},
                {"grade_result": {"sufficient": True}},
            ]
            metrics = await compute_grading_metrics(pairs)

        assert metrics["fnr"] < 0.3

    @pytest.mark.asyncio
    async def test_grader_accuracy_above_threshold(self):
        """Overall grading accuracy must be >= 0.7."""
        from evals.grading import compute_grading_metrics

        pairs = [
            {"query": "Q1", "chunks": [{"score": 0.8, "text": "t"}], "label": True},
            {"query": "Q2", "chunks": [{"score": 0.2, "text": "t"}], "label": False},
            {"query": "Q3", "chunks": [{"score": 0.9, "text": "t"}], "label": True},
            {"query": "Q4", "chunks": [{"score": 0.1, "text": "t"}], "label": False},
        ]

        with patch("evals.grading._run_grade_node", new_callable=AsyncMock) as mock_grade:
            mock_grade.side_effect = [
                {"grade_result": {"sufficient": True}},
                {"grade_result": {"sufficient": False}},
                {"grade_result": {"sufficient": True}},
                {"grade_result": {"sufficient": False}},
            ]
            metrics = await compute_grading_metrics(pairs)

        assert metrics["accuracy"] >= 0.7

    @pytest.mark.asyncio
    async def test_grading_metrics_has_required_keys(self):
        """compute_grading_metrics must return accuracy, fnr, fpr."""
        from evals.grading import compute_grading_metrics

        pairs = [
            {"query": "Q", "chunks": [{"score": 0.8, "text": "t"}], "label": True},
        ]

        with patch("evals.grading._run_grade_node", new_callable=AsyncMock) as mock_grade:
            mock_grade.return_value = {"grade_result": {"sufficient": True}}
            metrics = await compute_grading_metrics(pairs)

        assert "accuracy" in metrics
        assert "fnr" in metrics
        assert "fpr" in metrics

    def test_score_threshold_calibration_plot_generated(self, tmp_path):
        """plot_score_distribution must create a PNG file at the given path."""
        from evals.grading import plot_score_distribution

        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        output = tmp_path / "calibration.png"

        plot_score_distribution(scores, output)

        assert output.exists()
        assert output.stat().st_size > 0
