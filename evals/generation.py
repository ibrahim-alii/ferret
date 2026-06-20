"""RAGAS-based generation quality evaluation using Groq LLM + the project embedder.

Uses langchain_groq for the judge LLM and a thin adapter over the runtime embedder
(OpenAI or Gemini, per USE_LOCAL_EMBEDDINGS) for RAGAS's embedding needs.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import Any

from langchain_core.embeddings import Embeddings

from backend.ingestion.embedder import embed_chunks, embed_query


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


class _ProjectEmbeddings(Embeddings):
    """Adapter exposing the runtime embedder (OpenAI/Gemini) as a langchain Embeddings,
    so RAGAS scores with the same vectors the app retrieves with — no Voyage dependency."""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await embed_chunks([SimpleNamespace(text=t) for t in texts])

    async def aembed_query(self, text: str) -> list[float]:
        return await embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return asyncio.run(self.aembed_query(text))


def _run_ragas(dataset: list[dict[str, Any]]) -> dict[str, float]:
    """Run RAGAS evaluation configured with Groq LLM + the project embedder."""
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from langchain_groq import ChatGroq
    from datasets import Dataset

    llm = ChatGroq(
        model=os.environ.get("GENERATION_MODEL", "llama-3.3-70b-versatile"),
    )
    embeddings = _ProjectEmbeddings()

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
