"""Grading accuracy evaluation — run grade_node against labeled pairs."""
from __future__ import annotations

import os
from pathlib import Path
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


def plot_score_distribution(scores: list[float], output_path: Path) -> None:
    """Save a histogram of retrieval scores to show threshold band calibration."""
    import matplotlib.pyplot as plt

    high_threshold = float(os.environ.get("GRADE_HIGH_THRESHOLD", "0.6"))
    low_threshold = float(os.environ.get("GRADE_LOW_THRESHOLD", "0.35"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=20, color="steelblue", edgecolor="white")
    ax.axvline(high_threshold, color="green", linestyle="--", label=f"high={high_threshold}")
    ax.axvline(low_threshold, color="red", linestyle="--", label=f"low={low_threshold}")
    ax.set_xlabel("Top-chunk score")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution with threshold band")
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
