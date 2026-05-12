"""Classification + robustness metrics.

All functions take plain Python sequences / NumPy arrays so they can be used
both in cluster-side training scripts and in offline analysis notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    auroc: float
    precision: float
    recall: float
    fpr: float
    fnr: float

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "auroc": self.auroc,
            "precision": self.precision,
            "recall": self.recall,
            "fpr": self.fpr,
            "fnr": self.fnr,
        }


def classification_metrics(
    probs: Sequence[float],
    labels: Sequence[int],
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute metrics from probability outputs."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    preds = [1 if p >= threshold else 0 for p in probs]
    auroc = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan")
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    # cm = [[tn, fp], [fn, tp]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    return ClassificationMetrics(
        accuracy=float(accuracy_score(labels, preds)),
        macro_f1=float(f1_score(labels, preds, average="macro", zero_division=0)),
        auroc=auroc,
        precision=float(precision_score(labels, preds, zero_division=0)),
        recall=float(recall_score(labels, preds, zero_division=0)),
        fpr=float(fpr),
        fnr=float(fnr),
    )


def attack_success_rate(
    clean_probs: Sequence[float],
    attacked_probs: Sequence[float],
    labels: Sequence[int],
    threshold: float = 0.5,
) -> float:
    """Fraction of clean-correct examples that the attack flipped."""
    flipped = 0
    eligible = 0
    for c, a, y in zip(clean_probs, attacked_probs, labels):
        clean_pred = 1 if c >= threshold else 0
        attacked_pred = 1 if a >= threshold else 0
        if clean_pred == y:
            eligible += 1
            if attacked_pred != y:
                flipped += 1
    return flipped / eligible if eligible > 0 else float("nan")


def robustness_gap(clean: ClassificationMetrics, attacked: ClassificationMetrics) -> dict[str, float]:
    """Per-metric (clean - attacked) deltas."""
    out: dict[str, float] = {}
    for k, c in clean.to_dict().items():
        a = attacked.to_dict()[k]
        out[k] = float(c - a)
    return out
