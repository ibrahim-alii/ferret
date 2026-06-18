"""Tests for evals/baseline.py — regression detection."""
from __future__ import annotations

import json
import pytest


@pytest.mark.eval
class TestBaseline:
    def test_save_and_load_baseline(self, tmp_path):
        from evals.baseline import save_baseline, load_baseline

        scores = {"faithfulness": 0.85, "recall@5": 0.72}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("evals.baseline.REPORTS_DIR", tmp_path)
            save_baseline(scores, "paper123")
            loaded = load_baseline("paper123")

        assert loaded["faithfulness"] == pytest.approx(0.85)
        assert loaded["recall@5"] == pytest.approx(0.72)

    def test_no_regression_passes_when_scores_above_baseline(self, tmp_path):
        from evals.baseline import save_baseline, assert_no_regression

        baseline = {"faithfulness": 0.80, "recall@5": 0.70}
        current = {"faithfulness": 0.85, "recall@5": 0.75}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("evals.baseline.REPORTS_DIR", tmp_path)
            save_baseline(baseline, "paper_pass")
            assert_no_regression(current, "paper_pass", tolerance=0.05)

    def test_no_regression_fails_when_scores_below_baseline(self, tmp_path):
        from evals.baseline import save_baseline, assert_no_regression

        baseline = {"faithfulness": 0.85, "recall@5": 0.72}
        current = {"faithfulness": 0.50, "recall@5": 0.40}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("evals.baseline.REPORTS_DIR", tmp_path)
            save_baseline(baseline, "paper_fail")

            with pytest.raises(AssertionError):
                assert_no_regression(current, "paper_fail", tolerance=0.05)

    def test_eval_scores_do_not_regress_vs_baseline(self, tmp_path):
        """Scores at or above baseline minus tolerance should not raise."""
        from evals.baseline import save_baseline, assert_no_regression

        baseline = {"faithfulness": 0.80}
        current = {"faithfulness": 0.76}  # within 0.05 tolerance

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("evals.baseline.REPORTS_DIR", tmp_path)
            save_baseline(baseline, "paper_tol")
            assert_no_regression(current, "paper_tol", tolerance=0.05)
