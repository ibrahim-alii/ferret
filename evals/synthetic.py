"""Synthetic qrel and eval dataset generation, cached to disk."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import groq

CACHE_DIR = Path(__file__).parent / "cache"

_groq_client: groq.Groq | None = None


def _get_client() -> groq.Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = groq.Groq()
    return _groq_client


def _call_llm(prompt: str) -> str:
    model = os.environ.get("GRADING_MODEL", "llama-3.1-8b-instant")
    client = _get_client()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )
    return completion.choices[0].message.content.strip()


def _qrels_cache_path(paper_id: str) -> Path:
    return Path(CACHE_DIR) / f"{paper_id}_qrels.json"


def _dataset_cache_path(paper_id: str, mode: str) -> Path:
    return Path(CACHE_DIR) / f"{paper_id}_dataset_{mode}.json"


def generate_qrels(paper_id: str) -> dict[str, dict[str, int]]:
    """Generate relevance judgments for a paper, cached to disk.

    Returns: {query_id: {chunk_id: relevance_score (0-2)}}
    """
    cache_path = _qrels_cache_path(paper_id)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    prompt = (
        f"For paper '{paper_id}', generate 5 evaluation queries and their relevant chunk IDs. "
        "Return JSON array: [{\"query\": \"...\", \"relevant_chunk_ids\": [\"c1\", \"c2\"]}]. "
        "Use realistic chunk IDs like 'chunk_001', 'chunk_002', etc."
    )
    raw = _call_llm(prompt)

    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        entries = json.loads(raw[start:end])

    qrels: dict[str, dict[str, int]] = {}
    for entry in entries:
        query_id = entry["query"]
        chunk_map: dict[str, int] = {}
        for cid in entry.get("relevant_chunk_ids", []):
            chunk_map[cid] = 1
        qrels[query_id] = chunk_map

    with open(cache_path, "w") as f:
        json.dump(qrels, f, indent=2)

    return qrels


def generate_eval_dataset(paper_id: str, mode: str) -> list[dict[str, Any]]:
    """Generate RAGAS-format eval dataset for a paper+mode, cached to disk.

    Returns: [{question, answer, contexts}]
    """
    cache_path = _dataset_cache_path(paper_id, mode)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    prompt = (
        f"For paper '{paper_id}' in mode '{mode}', generate 5 QA pairs with context. "
        "Return JSON array: [{\"question\": \"...\", \"answer\": \"...\", \"contexts\": [\"...\"]}]."
    )
    raw = _call_llm(prompt)

    try:
        dataset = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        dataset = json.loads(raw[start:end])

    with open(cache_path, "w") as f:
        json.dump(dataset, f, indent=2)

    return dataset
