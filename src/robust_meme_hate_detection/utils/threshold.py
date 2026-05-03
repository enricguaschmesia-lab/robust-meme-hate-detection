"""Decision-threshold sweep on validation predictions.

Picks the threshold τ in a fine grid that maximises macro-F1 on a labelled
split. Produces both the chosen τ and a small CSV-style table for inspection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    macro_f1: float
    accuracy: float
    precision: float
    recall: float

    def to_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
        }


def sweep_threshold(
    probs: Sequence[float],
    labels: Sequence[int],
    grid: Iterable[float] | None = None,
) -> tuple[ThresholdResult, list[ThresholdResult]]:
    """Return (best_by_macro_f1, all_grid_points)."""

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    if grid is None:
        grid = [round(0.30 + 0.01 * i, 2) for i in range(41)]  # 0.30 .. 0.70

    grid_results: list[ThresholdResult] = []
    for tau in grid:
        preds = [1 if p >= tau else 0 for p in probs]
        result = ThresholdResult(
            threshold=float(tau),
            macro_f1=float(f1_score(labels, preds, average="macro", zero_division=0)),
            accuracy=float(accuracy_score(labels, preds)),
            precision=float(precision_score(labels, preds, zero_division=0)),
            recall=float(recall_score(labels, preds, zero_division=0)),
        )
        grid_results.append(result)

    best = max(grid_results, key=lambda r: (r.macro_f1, r.accuracy))
    return best, grid_results


def persist_threshold(out_path: str | Path, best: ThresholdResult, grid: list[ThresholdResult]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best": best.to_dict(),
        "grid": [r.to_dict() for r in grid],
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path
