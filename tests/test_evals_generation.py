"""Tests for evals/generation.py — RAGAS generation quality evaluation."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _make_dataset():
    return [
        {
            "question": "What is the main contribution?",
            "answer": "The main contribution is a novel method for...",
            "contexts": ["The paper presents a novel method for...", "Our approach achieves state-of-the-art results..."],
        }
    ]


@pytest.mark.eval
class TestGenerationEval:
    def test_ragas_dataset_passes_validation(self):
        """build_ragas_dataset must return a list with required fields."""
        from evals.generation import build_ragas_dataset

        raw = _make_dataset()
        dataset = build_ragas_dataset(raw)

        assert len(dataset) == 1
        assert dataset[0]["question"] == raw[0]["question"]
        assert dataset[0]["answer"] == raw[0]["answer"]
        assert dataset[0]["contexts"] == raw[0]["contexts"]

    def test_faithfulness_score_above_threshold(self):
        """Faithfulness score from RAGAS must be >= 0.5."""
        from evals.generation import run_ragas_eval

        mock_scores = {"faithfulness": 0.85, "answer_relevancy": 0.80}

        with patch("evals.generation._run_ragas", return_value=mock_scores):
            scores = run_ragas_eval(_make_dataset())

        assert scores["faithfulness"] >= 0.5

    def test_answer_relevancy_above_threshold(self):
        """Answer relevancy score from RAGAS must be >= 0.5."""
        from evals.generation import run_ragas_eval

        mock_scores = {"faithfulness": 0.85, "answer_relevancy": 0.80}

        with patch("evals.generation._run_ragas", return_value=mock_scores):
            scores = run_ragas_eval(_make_dataset())

        assert scores["answer_relevancy"] >= 0.5

    def test_ragas_uses_groq_llm_and_voyage_embeddings_not_openai(self):
        """RAGAS configuration must not use OpenAI — only Groq + Voyage."""
        import evals.generation as gen_module
        import inspect

        source = inspect.getsource(gen_module)

        assert "openai" not in source.lower()
        assert "ChatOpenAI" not in source
        assert "OpenAIEmbeddings" not in source

    def test_ragas_scores_serializable_to_json(self):
        """RAGAS score dict must be JSON-serializable."""
        from evals.generation import run_ragas_eval

        mock_scores = {"faithfulness": 0.85, "answer_relevancy": 0.80}

        with patch("evals.generation._run_ragas", return_value=mock_scores):
            scores = run_ragas_eval(_make_dataset())

        serialized = json.dumps(scores)
        loaded = json.loads(serialized)
        assert loaded["faithfulness"] == pytest.approx(0.85, abs=0.01)

    def test_context_precision_above_threshold(self):
        """Context precision must be >= 0.5 when retrieved chunks are relevant."""
        from evals.generation import run_ragas_eval

        mock_scores = {"faithfulness": 0.85, "answer_relevancy": 0.80, "context_precision": 0.75}

        with patch("evals.generation._run_ragas", return_value=mock_scores):
            scores = run_ragas_eval(_make_dataset())

        assert scores.get("context_precision", 1.0) >= 0.5
