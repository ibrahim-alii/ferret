"""Baseline save/load and regression assertion."""
from __future__ import annotations

import json
from pathlib import Path

REPORTS_DIR = Path(__file__).parent / "reports"


def _baseline_path(paper_id: str) -> Path:
    return Path(REPORTS_DIR) / paper_id / "baseline.json"


def save_baseline(scores: dict[str, float], paper_id: str) -> None:
    path = _baseline_path(paper_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(scores, f, indent=2)


def load_baseline(paper_id: str) -> dict[str, float]:
    path = _baseline_path(paper_id)
    with open(path) as f:
        return json.load(f)


def assert_no_regression(
    current_scores: dict[str, float],
    paper_id: str,
    tolerance: float = 0.05,
) -> None:
    """Raise AssertionError if any score in current_scores is below baseline - tolerance."""
    baseline = load_baseline(paper_id)
    failures: list[str] = []

    for metric, baseline_val in baseline.items():
        if metric not in current_scores:
            continue
        current_val = current_scores[metric]
        if current_val < baseline_val - tolerance:
            failures.append(
                f"{metric}: current={current_val:.4f} < baseline={baseline_val:.4f} - tolerance={tolerance}"
            )

    if failures:
        raise AssertionError("Eval regression detected:\n" + "\n".join(failures))
