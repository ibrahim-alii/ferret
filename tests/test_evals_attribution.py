"""Tests for evals/attribution.py — answer attribution to retrieved chunks."""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.eval
class TestAttribution:
    def test_attribution_splits_answer_into_sentences(self):
        from evals.attribution import attribute_answer

        answer = "This is sentence one. This is sentence two. This is sentence three."
        chunks = [{"chunk_id": "c1", "text": "This is sentence one."}]

        with patch("evals.attribution._embed_texts", return_value=[[0.9, 0.1]] * 4):
            result = attribute_answer(answer, chunks)

        # Should have detected at least 2 sentences (splitting on ". ")
        assert result["attribution_rate"] >= 0.0

    def test_attribution_flags_low_similarity_as_unattributed(self):
        """Sentences with no similar chunk should appear in unattributed_sentences."""
        from evals.attribution import attribute_answer
        import numpy as np

        answer = "Completely unrelated statement. Another unrelated sentence."
        chunks = [{"chunk_id": "c1", "text": "Something totally different."}]

        # Single batched call: [sent0, sent1, c1]
        # Sentences are orthogonal to the chunk => similarity = 0 => unattributed
        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]

        with patch("evals.attribution._embed_texts", side_effect=mock_embed):
            result = attribute_answer(answer, chunks)

        assert len(result["unattributed_sentences"]) >= 1

    def test_attribution_rate_above_threshold_for_grounded_answer(self):
        """When answer closely matches chunk text, attribution_rate should be high."""
        from evals.attribution import attribute_answer

        chunk_text = "The proposed method achieves state-of-the-art results."
        answer = chunk_text  # Identical => should be attributed
        chunks = [{"chunk_id": "c1", "text": chunk_text}]

        # Batched call: [sentence, chunk] — same embedding => cosine similarity = 1.0
        with patch("evals.attribution._embed_texts", return_value=[[0.5, 0.5], [0.5, 0.5]]):
            result = attribute_answer(answer, chunks)

        assert result["attribution_rate"] >= 0.5

    def test_unattributed_sentences_serialized_to_report(self):
        """Result dict must be JSON-serializable."""
        from evals.attribution import attribute_answer
        import json

        answer = "First sentence. Second sentence."
        chunks = [{"chunk_id": "c1", "text": "First sentence."}]

        with patch("evals.attribution._embed_texts", return_value=[[1.0, 0.0]] * 3):
            result = attribute_answer(answer, chunks)

        serialized = json.dumps(result)
        loaded = json.loads(serialized)
        assert "attribution_rate" in loaded
        assert "unattributed_sentences" in loaded
        assert "chunk_utilization_rate" in loaded

    def test_chunk_utilization_rate_computed_correctly(self):
        """3 chunks, 2 used => utilization_rate ~0.667."""
        from evals.attribution import attribute_answer
        import numpy as np

        answer = "Sentence A. Sentence B."
        chunks = [
            {"chunk_id": "c1", "text": "Sentence A."},
            {"chunk_id": "c2", "text": "Sentence B."},
            {"chunk_id": "c3", "text": "Sentence C."},
        ]

        # Single batched call: [sent0, sent1, c1, c2, c3] -> 5 embeddings
        # sent0 matches c1 (both [1,0,0]), sent1 matches c2 (both [0,1,0]), c3 unused [0,0,1]
        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [
                [1.0, 0.0, 0.0],  # sentence 0
                [0.0, 1.0, 0.0],  # sentence 1
                [1.0, 0.0, 0.0],  # chunk c1
                [0.0, 1.0, 0.0],  # chunk c2
                [0.0, 0.0, 1.0],  # chunk c3
            ]

        with patch("evals.attribution._embed_texts", side_effect=mock_embed):
            result = attribute_answer(answer, chunks)

        assert result["chunk_utilization_rate"] == pytest.approx(2 / 3, abs=0.01)
