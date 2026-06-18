"""Ranx-based retrieval evaluation: build Qrels/Run objects and compute metrics."""
from __future__ import annotations

from typing import Any

from ranx import Qrels, Run, evaluate, compare


def build_run(results: dict[str, list[dict]], run_name: str = "run") -> Run:
    """Convert {query_id: [{chunk_id, score, ...}]} to a ranx Run."""
    run_dict: dict[str, dict[str, float]] = {}
    for query_id, chunks in results.items():
        run_dict[query_id] = {c["chunk_id"]: float(c["score"]) for c in chunks}
    return Run(run_dict, name=run_name)


def compute_metrics(
    qrels_dict: dict[str, dict[str, int]],
    run_dict: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute recall@5, ndcg@5, mrr for a single run."""
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    metrics = evaluate(
        qrels,
        run,
        ["recall@5", "ndcg@5", "mrr"],
    )
    return {k: float(v) for k, v in metrics.items()}


def build_comparison_report(
    qrels_dict: dict[str, dict[str, int]],
    runs_dict: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    """Compare multiple runs (dense / hybrid / hybrid+rerank) and return a JSON-safe report."""
    qrels = Qrels(qrels_dict)
    report: dict[str, Any] = {"runs": {}}

    for run_name, run_data in runs_dict.items():
        run = Run(run_data, name=run_name)
        metrics = evaluate(qrels, run, ["recall@5", "ndcg@5", "mrr"])
        report["runs"][run_name] = {k: float(v) for k, v in metrics.items()}

    return report
