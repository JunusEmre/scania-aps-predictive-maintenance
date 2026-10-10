"""Evaluation helpers for binary APS classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

FALSE_POSITIVE_COST = 10
FALSE_NEGATIVE_COST = 500


def confusion_counts(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
) -> dict[str, int]:
    """Return TN, FP, FN, TP for binary labels encoded as 0 (neg) and 1 (pos)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def scania_business_cost(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    false_positive_cost: int = FALSE_POSITIVE_COST,
    false_negative_cost: int = FALSE_NEGATIVE_COST,
) -> int:
    """Compute Scania cost: FP * 10 + FN * 500."""
    counts = confusion_counts(y_true, y_pred)
    return (
        counts["fp"] * false_positive_cost
        + counts["fn"] * false_negative_cost
    )


def evaluate_binary_classifier(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_score: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """Summarise common metrics for internal validation."""
    counts = confusion_counts(y_true, y_pred)
    results: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        **counts,
        "business_cost": scania_business_cost(y_true, y_pred),
    }
    if y_score is not None:
        results["pr_auc"] = float(average_precision_score(y_true, y_score))
    return results
