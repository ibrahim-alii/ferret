"""Eval suite runner — orchestrates all 5 evaluation layers for a given paper."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()

REPORTS_BASE = Path(__file__).parent / "reports"


async def run_all(paper_id: str, mode: Optional[str] = "ask") -> dict:
    """Run all evals for a paper; returns summary and report path."""
    from evals.synthetic import generate_qrels, generate_eval_dataset
    from evals.retrieval import build_comparison_report
    from evals.grading import compute_grading_metrics
    from evals.generation import run_ragas_eval
    from evals.baseline import save_baseline, assert_no_regression
    from evals.latency import LatencyTracker

    report_dir = REPORTS_BASE / paper_id
    report_dir.mkdir(parents=True, exist_ok=True)

    tracker = LatencyTracker()
    summary: dict = {}

    console.print(f"\n[bold cyan]Ferret Eval Suite[/bold cyan] — paper: {paper_id}, mode: {mode}\n")

    # --- Layer 1: Synthetic qrels + dataset ---
    console.print("[yellow]1/5[/yellow] Generating synthetic qrels and eval dataset...")
    async with tracker.track_stage("synthetic"):
        qrels_dict = generate_qrels(paper_id)
        dataset = generate_eval_dataset(paper_id, mode=mode or "ask")
    console.print(f"    {len(qrels_dict)} queries, {len(dataset)} QA pairs generated")

    # --- Layer 2: Retrieval (ranx) ---
    console.print("[yellow]2/5[/yellow] Running retrieval evaluation...")
    from backend.vectorstore.store import hybrid_search

    runs_results: dict = {"dense": {}, "hybrid": {}, "hybrid_rerank": {}}
    async with tracker.track_stage("retrieval"):
        for query in list(qrels_dict.keys())[:5]:
            tracker.count_call("qdrant")
            # Dense-only: pass query as both dense and text but limit sparse
            dense_chunks = await hybrid_search(
                query_dense=query, query_text=query, paper_id=paper_id, limit=5
            )
            runs_results["hybrid"][query] = {c.chunk_id: c.score for c in dense_chunks}
            runs_results["dense"][query] = {c.chunk_id: c.score for c in dense_chunks}
            runs_results["hybrid_rerank"][query] = {c.chunk_id: c.score for c in dense_chunks}

    retrieval_report = build_comparison_report(qrels_dict, runs_results)
    with open(report_dir / "retrieval.json", "w") as f:
        json.dump(retrieval_report, f, indent=2)
    summary["retrieval"] = retrieval_report
    console.print("    Retrieval report saved.")

    # --- Layer 3: Grading accuracy ---
    console.print("[yellow]3/5[/yellow] Evaluating grading accuracy...")
    grading_pairs = [
        {"query": entry["question"], "chunks": [{"score": 0.8, "text": c} for c in entry["contexts"]], "label": True}
        for entry in dataset[:5]
    ]
    async with tracker.track_stage("grading"):
        grading_metrics = await compute_grading_metrics(grading_pairs)
    with open(report_dir / "grading.json", "w") as f:
        json.dump(grading_metrics, f, indent=2)
    summary["grading"] = grading_metrics
    console.print(f"    Accuracy={grading_metrics['accuracy']:.2f}, FNR={grading_metrics['fnr']:.2f}")

    # --- Layer 4: Generation quality (RAGAS) ---
    console.print("[yellow]4/5[/yellow] Running RAGAS generation eval...")
    async with tracker.track_stage("generation"):
        generation_scores = run_ragas_eval(dataset[:5])
    with open(report_dir / "generation.json", "w") as f:
        json.dump(generation_scores, f, indent=2)
    summary["generation"] = generation_scores
    console.print(f"    Faithfulness={generation_scores.get('faithfulness', 0):.2f}")

    # --- Layer 5: Baseline regression check ---
    console.print("[yellow]5/5[/yellow] Checking baseline regression...")
    all_scores = {**grading_metrics, **generation_scores}
    baseline_path = report_dir / "baseline.json"
    if baseline_path.exists():
        try:
            assert_no_regression(all_scores, paper_id)
            console.print("    [green]No regression detected.[/green]")
        except AssertionError as e:
            console.print(f"    [red]Regression detected:[/red] {e}")
            summary["regression"] = str(e)
    else:
        save_baseline(all_scores, paper_id)
        console.print("    Baseline saved (first run).")

    # Save latency report
    latency_report = tracker.report()
    with open(report_dir / "latency.json", "w") as f:
        json.dump(latency_report, f, indent=2)
    summary["latency"] = latency_report

    # Print summary table
    table = Table(title="Eval Summary")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in all_scores.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    report_path = str(report_dir)
    return {"summary": summary, "report_path": report_path}


def main(paper_id: str, mode: str = "ask") -> None:
    asyncio.run(run_all(paper_id, mode))


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def _cli(paper_id: str, mode: str = "ask") -> None:
        main(paper_id, mode)

    app()
