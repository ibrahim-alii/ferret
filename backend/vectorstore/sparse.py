"""Single BM25 encoder instance shared by upsert and query paths."""
from __future__ import annotations

import os
import threading

from fastembed import SparseTextEmbedding

_encoder: SparseTextEmbedding | None = None
_encoder_lock = threading.Lock()


def get_sparse_encoder() -> SparseTextEmbedding:
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                model_name = os.environ.get("SPARSE_MODEL", "Qdrant/bm25")
                _encoder = SparseTextEmbedding(model_name=model_name)
    return _encoder


def encode_query(text: str):
    enc = get_sparse_encoder()
    results = list(enc.query_embed(text))
    if not results:
        raise ValueError(f"Sparse encoder returned no results for query: {text!r}")
    return results[0]


def encode_document(text: str):
    enc = get_sparse_encoder()
    results = list(enc.passage_embed(text))
    if not results:
        raise ValueError(f"Sparse encoder returned no results for document: {text!r}")
    return results[0]
