"""Tests for baseline model pipelines."""

import numpy as np
import pandas as pd

from src.models.baseline import (
    CONSTANT_FEATURE_TO_DROP,
    build_logistic_regression_pipeline,
)


def _synthetic_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, np.nan, 4.0],
            "feat_b": [0.0, 1.0, 2.0, 3.0],
            CONSTANT_FEATURE_TO_DROP: [5.0, 5.0, 5.0, 5.0],
        }
    )


def test_logistic_regression_pipeline_fits_with_missing_values():
    X = _synthetic_features()
    y = np.array([0, 0, 1, 1])
    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert preds.shape == (4,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_pipeline_predictions_are_binary():
    X = _synthetic_features()
    y = np.array([0, 1, 0, 1])
    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert np.all((preds == 0) | (preds == 1))
