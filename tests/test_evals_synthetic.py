"""Tests for evals/synthetic.py — qrel and eval dataset generation."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest


@pytest.mark.eval
class TestQrelGeneration:
    def test_qrel_generation_produces_valid_ranx_format(self, tmp_path):
        """qrels must be {query_id: {chunk_id: relevance_score}} with int scores."""
        from evals.synthetic import generate_qrels

        llm_content = json.dumps(
            [{"query": "What is the method?", "relevant_chunk_ids": ["c1", "c2"]}]
        )

        with (
            patch("evals.synthetic.CACHE_DIR", tmp_path),
            patch("evals.synthetic._call_llm", return_value=llm_content),
        ):
            qrels = generate_qrels("paper123")

        assert isinstance(qrels, dict)
        for query_id, chunk_map in qrels.items():
            assert isinstance(query_id, str)
            assert isinstance(chunk_map, dict)
            for chunk_id, score in chunk_map.items():
                assert isinstance(chunk_id, str)
                assert isinstance(score, int)
                assert score in (0, 1, 2)

    def test_qrel_generation_is_cached_on_second_call(self, tmp_path):
        """LLM should only be called once; second call reads from cache."""
        from evals.synthetic import generate_qrels

        llm_content = json.dumps(
            [{"query": "What is X?", "relevant_chunk_ids": ["c1"]}]
        )
        call_count = {"n": 0}

        def fake_call_llm(prompt: str) -> str:
            call_count["n"] += 1
            return llm_content

        with (
            patch("evals.synthetic.CACHE_DIR", tmp_path),
            patch("evals.synthetic._call_llm", side_effect=fake_call_llm),
        ):
            generate_qrels("paper_abc")
            generate_qrels("paper_abc")

        assert call_count["n"] == 1

    def test_eval_dataset_generation_produces_ragas_format(self, tmp_path):
        """Dataset entries must have question, answer, and contexts fields."""
        from evals.synthetic import generate_eval_dataset

        llm_content = json.dumps(
            [
                {
                    "question": "What is the contribution?",
                    "answer": "The main contribution is...",
                    "contexts": ["chunk text A", "chunk text B"],
                }
            ]
        )

        with (
            patch("evals.synthetic.CACHE_DIR", tmp_path),
            patch("evals.synthetic._call_llm", return_value=llm_content),
        ):
            dataset = generate_eval_dataset("paper123", mode="ask")

        assert isinstance(dataset, list)
        assert len(dataset) > 0
        for entry in dataset:
            assert "question" in entry
            assert "answer" in entry
            assert "contexts" in entry
            assert isinstance(entry["contexts"], list)

    def test_eval_dataset_cached_on_second_call(self, tmp_path):
        """Dataset generation LLM called once; second call is a cache hit."""
        from evals.synthetic import generate_eval_dataset

        llm_content = json.dumps(
            [{"question": "Q?", "answer": "A.", "contexts": ["c"]}]
        )
        call_count = {"n": 0}

        def fake_call(prompt: str) -> str:
            call_count["n"] += 1
            return llm_content

        with (
            patch("evals.synthetic.CACHE_DIR", tmp_path),
            patch("evals.synthetic._call_llm", side_effect=fake_call),
        ):
            generate_eval_dataset("paper_xyz", mode="deep_dive")
            generate_eval_dataset("paper_xyz", mode="deep_dive")

        assert call_count["n"] == 1
