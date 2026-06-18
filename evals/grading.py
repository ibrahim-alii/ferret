"""Grading accuracy evaluation — run grade_node against labeled pairs."""
from __future__ import annotations

from typing import Any


async def _run_grade_node(state: dict) -> dict:
    """Thin wrapper so tests can patch it without importing the full graph."""
    from backend.graph.nodes.grade import grade_node

    return await grade_node(state)


async def compute_grading_metrics(
    pairs: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute accuracy, FNR, FPR for a list of labeled (query, chunks, label) pairs.

    Each pair: {query: str, chunks: [{score, text}], label: bool}
    """
    tp = fp = tn = fn = 0

    for pair in pairs:
        state = {
            "user_message": pair["query"],
            "reranked_children": pair["chunks"],
        }
        result = await _run_grade_node(state)
        predicted = result["grade_result"]["sufficient"]
        actual = pair["label"]

        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {"accuracy": accuracy, "fnr": fnr, "fpr": fpr}
