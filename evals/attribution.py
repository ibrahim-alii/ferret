"""Answer attribution: map answer sentences to retrieved chunks via embedding similarity."""
from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import voyageai


_SIMILARITY_THRESHOLD = 0.75


def _embed_texts(texts: list[str]) -> list[list[float]]:
    client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY", ""))
    result = client.embed(texts, model="voyage-3", input_type="document")
    return result.embeddings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def attribute_answer(
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
    threshold: float = _SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """Split answer into sentences and attribute each to the most similar chunk.

    Returns: {attribution_rate, unattributed_sentences, chunk_utilization_rate}
    """
    sentences = _split_sentences(answer)
    chunk_texts = [c["text"] for c in retrieved_chunks]
    chunk_ids = [c["chunk_id"] for c in retrieved_chunks]

    sentence_embeddings = _embed_texts(sentences)
    chunk_embeddings = _embed_texts(chunk_texts)

    unattributed: list[str] = []
    used_chunk_ids: set[str] = set()

    for i, sent_emb in enumerate(sentence_embeddings):
        best_sim = 0.0
        best_cid: str | None = None
        for j, chunk_emb in enumerate(chunk_embeddings):
            sim = _cosine_similarity(sent_emb, chunk_emb)
            if sim > best_sim:
                best_sim = sim
                best_cid = chunk_ids[j]

        if best_sim >= threshold and best_cid is not None:
            used_chunk_ids.add(best_cid)
        else:
            unattributed.append(sentences[i])

    total = len(sentences)
    attribution_rate = (total - len(unattributed)) / total if total else 0.0
    chunk_utilization_rate = len(used_chunk_ids) / len(chunk_ids) if chunk_ids else 0.0

    return {
        "attribution_rate": attribution_rate,
        "unattributed_sentences": unattributed,
        "chunk_utilization_rate": chunk_utilization_rate,
    }
