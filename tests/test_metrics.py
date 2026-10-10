"""Tests for Scania evaluation metrics."""

import numpy as np

from src.evaluation.metrics import (
    confusion_counts,
    evaluate_binary_classifier,
    scania_business_cost,
)


def test_scania_business_cost_hand_checkable():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]
    counts = confusion_counts(y_true, y_pred)
    assert counts == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert scania_business_cost(y_true, y_pred) == 1 * 10 + 1 * 500


def test_evaluate_binary_classifier_includes_pr_auc_when_scores_given():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    results = evaluate_binary_classifier(y_true, y_pred, y_score=y_score)
    assert results["fp"] == 0
    assert results["fn"] == 0
    assert results["business_cost"] == 0
    assert 0.0 <= results["pr_auc"] <= 1.0
