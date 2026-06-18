"""Eval suite runner — orchestrates all 5 evaluation layers for a given paper."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()

REPORTS_BASE = Path(__file__).parent / "reports"


async def run_all(paper_id: str, mode: str | None = "ask") -> dict[str, Any]:
    """Run all evals for a paper; returns summary and report path."""
    from evals.synthetic import generate_qrels, generate_eval_dataset
    from evals.retrieval import build_comparison_report
    from evals.grading import compute_grading_metrics, plot_score_distribution
    from evals.generation import run_ragas_eval
    from evals.attribution import attribute_answer
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

    # --- Layer 2: Retrieval (ranx) — three distinct strategies ---
    console.print("[yellow]2/5[/yellow] Running retrieval evaluation (dense / hybrid / hybrid+rerank)...")
    from backend.vectorstore.store import hybrid_search
    from backend.graph.nodes.rerank import rerank_node

    runs_results: dict = {"dense": {}, "hybrid": {}, "hybrid_rerank": {}}
    # NOTE: zero vectors are used as query embeddings to avoid Voyage API calls during eval.
    # The hybrid and hybrid+rerank runs are compared against each other relative to qrels,
    # so consistent (if meaningless) dense vectors still measure the *reranking* lift correctly.
    # The dense-only run will reflect cosine-nearest to the zero vector (arbitrary), so its
    # absolute Recall@k numbers should be interpreted as a lower-bound baseline, not a fair
    # dense-retrieval measurement. For a fair dense comparison, replace dummy_dense with real
    # Voyage embeddings per query.
    eval_queries = list(qrels_dict.keys())[:5]
    embed_dim = int(os.environ.get("VOYAGE_EMBED_DIM", "1024"))
    dummy_dense = [0.0] * embed_dim

    async with tracker.track_stage("retrieval"):
        for query in eval_queries:
            tracker.count_call("qdrant")

            dense_chunks = await hybrid_search(
                query_dense=dummy_dense,
                query_text=query,
                paper_id=paper_id,
                limit=5,
                use_sparse=False,
            )
            runs_results["dense"][query] = {str(c.chunk_id): c.score for c in dense_chunks}

            tracker.count_call("qdrant")
            hybrid_chunks = await hybrid_search(
                query_dense=dummy_dense,
                query_text=query,
                paper_id=paper_id,
                limit=5,
                use_sparse=True,
            )
            runs_results["hybrid"][query] = {str(c.chunk_id): c.score for c in hybrid_chunks}

            # Rerank the hybrid candidates
            tracker.count_call("voyage")
            rerank_state = {
                "user_message": query,
                "retrieved_chunks": [
                    {"chunk_id": c.chunk_id, "score": c.score, "text": "", "section_name": c.section_name}
                    for c in hybrid_chunks
                ],
            }
            reranked_state = await rerank_node(rerank_state)
            reranked = reranked_state.get("reranked_children", [])
            runs_results["hybrid_rerank"][query] = {
                str(c["chunk_id"]): c.get("score", 0.0) for c in reranked
            }

    retrieval_report = build_comparison_report(qrels_dict, runs_results)
    with open(report_dir / "retrieval.json", "w") as f:
        json.dump(retrieval_report, f, indent=2)
    summary["retrieval"] = retrieval_report
    console.print("    Retrieval report saved.")

    # --- Layer 3: Grading accuracy + calibration plot ---
    console.print("[yellow]3/5[/yellow] Evaluating grading accuracy...")
    # Use actual top-chunk scores from the hybrid retrieval run for calibration;
    # fall back to 0.8 only when retrieval returned no results for a query.
    grading_pairs: list[dict] = []
    for entry in dataset[:5]:
        query = entry["question"]
        top_score = max(
            runs_results["hybrid"].get(query, {}).values(), default=0.8
        )
        grading_pairs.append({
            "query": query,
            "chunks": [{"score": top_score, "text": c} for c in entry["contexts"]],
            "label": True,
        })

    async with tracker.track_stage("grading"):
        grading_metrics = await compute_grading_metrics(grading_pairs)

    with open(report_dir / "grading.json", "w") as f:
        json.dump(grading_metrics, f, indent=2)
    summary["grading"] = grading_metrics
    console.print(f"    Accuracy={grading_metrics['accuracy']:.2f}, FNR={grading_metrics['fnr']:.2f}")

    # Calibration plot — uses actual top-chunk scores from hybrid retrieval
    grader_scores = [pair["chunks"][0]["score"] for pair in grading_pairs if pair["chunks"]]
    plot_score_distribution(grader_scores, report_dir / "calibration.png")
    console.print("    Calibration plot saved.")

    # --- Layer 4: Generation quality (RAGAS) ---
    console.print("[yellow]4/5[/yellow] Running RAGAS generation eval...")
    async with tracker.track_stage("generation"):
        generation_scores = run_ragas_eval(dataset[:5])
    with open(report_dir / "generation.json", "w") as f:
        json.dump(generation_scores, f, indent=2)
    summary["generation"] = generation_scores
    console.print(f"    Faithfulness={generation_scores.get('faithfulness', 0):.2f}")

    # --- Attribution: map answer sentences to retrieved chunks ---
    console.print("[yellow]+[/yellow] Running chunk-level attribution...")
    attribution_results: list[dict] = []
    async with tracker.track_stage("attribution"):
        for entry in dataset[:5]:
            answer = entry.get("answer", "")
            contexts = entry.get("contexts", [])
            if not answer or not contexts:
                continue
            tracker.count_call("voyage")
            chunks = [{"chunk_id": f"ctx_{i}", "text": c} for i, c in enumerate(contexts)]
            result = attribute_answer(answer, chunks)
            attribution_results.append(result)

    if attribution_results:
        avg_attr_rate = sum(r["attribution_rate"] for r in attribution_results) / len(attribution_results)
        avg_util_rate = sum(r["chunk_utilization_rate"] for r in attribution_results) / len(attribution_results)
    else:
        avg_attr_rate = avg_util_rate = 0.0

    attribution_report = {
        "attribution_rate": avg_attr_rate,
        "chunk_utilization_rate": avg_util_rate,
        "details": attribution_results,
    }
    with open(report_dir / "attribution.json", "w") as f:
        json.dump(attribution_report, f, indent=2)
    summary["attribution"] = attribution_report
    console.print(f"    Attribution rate={avg_attr_rate:.2f}, Chunk utilization={avg_util_rate:.2f}")

    # --- Layer 5: Baseline regression check ---
    console.print("[yellow]5/5[/yellow] Checking baseline regression...")
    all_scores = {
        **grading_metrics,
        **generation_scores,
        "attribution_rate": avg_attr_rate,
        "chunk_utilization_rate": avg_util_rate,
    }
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
