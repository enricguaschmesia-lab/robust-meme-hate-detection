"""Classification and robustness metrics.

AUROC is the primary, threshold-free metric; macro-F1 / accuracy / FPR / FNR are
reported at a fixed threshold (moderation has asymmetric error costs). The
attack-success rate (ASR) measures how many clean-correct predictions an attack
flips, and :func:`robustness_gap` is the per-metric clean-minus-attacked drop.
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
    """Compute accuracy / macro-F1 / AUROC / precision / recall / FPR / FNR."""
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
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(labels, preds, labels=[0, 1]).ravel())
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
    """Fraction of originally-correct predictions the attack flipped."""
    flipped = eligible = 0
    for c, a, y in zip(clean_probs, attacked_probs, labels):
        if (1 if c >= threshold else 0) == y:
            eligible += 1
            if (1 if a >= threshold else 0) != y:
                flipped += 1
    return flipped / eligible if eligible > 0 else float("nan")


def robustness_gap(clean: ClassificationMetrics, attacked: ClassificationMetrics) -> dict[str, float]:
    """Per-metric ``clean - attacked`` deltas (e.g. the AUROC robustness gap)."""
    attacked_d = attacked.to_dict()
    return {k: float(v - attacked_d[k]) for k, v in clean.to_dict().items()}
