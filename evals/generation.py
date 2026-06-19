"""RAGAS-based generation quality evaluation using Groq LLM + Voyage embeddings.

Configured with langchain_groq and langchain_community VoyageAIEmbeddings.
"""
from __future__ import annotations

import os
from typing import Any


def build_ragas_dataset(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and return RAGAS-format dataset entries."""
    return [
        {
            "question": entry["question"],
            "answer": entry["answer"],
            "contexts": entry["contexts"],
        }
        for entry in raw
    ]


def _run_ragas(dataset: list[dict[str, Any]]) -> dict[str, float]:
    """Run RAGAS evaluation configured with Groq + Voyage."""
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import VoyageAIEmbeddings
    from datasets import Dataset

    llm = ChatGroq(
        model=os.environ.get("GENERATION_MODEL", "llama-3.3-70b-versatile"),
    )
    voyage_key = os.environ.get("VOYAGE_API_KEY")
    if not voyage_key:
        raise EnvironmentError("VOYAGE_API_KEY is not set")
    embeddings = VoyageAIEmbeddings(
        voyage_api_key=voyage_key,
        model="voyage-3",
    )

    hf_dataset = Dataset.from_list(dataset)
    result = ragas_evaluate(
        hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
    )
    return {k: float(v) for k, v in result.items()}


def run_ragas_eval(dataset: list[dict[str, Any]]) -> dict[str, float]:
    """Build dataset and run RAGAS evaluation."""
    validated = build_ragas_dataset(dataset)
    return _run_ragas(validated)
